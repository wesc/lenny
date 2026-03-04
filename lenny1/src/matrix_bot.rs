use anyhow::{Result, bail};
use futures_util::StreamExt;
use matrix_sdk::{
    Client, SlidingSyncList, SlidingSyncMode,
    authentication::matrix::MatrixSession,
    room::Room,
    ruma::{
        OwnedUserId,
        events::{
            reaction::OriginalSyncReactionEvent,
            relation::Thread,
            room::member::StrippedRoomMemberEvent,
            room::message::{
                MessageType, OriginalSyncRoomMessageEvent, Relation, RoomMessageEventContent,
            },
        },
    },
    sliding_sync::Version,
};
use serde_json::json;
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use url::Url;

use crate::config::Config;
use crate::{cli_bot, once};

fn sanitize_room_name(name: &str) -> String {
    let s: String = name
        .to_lowercase()
        .chars()
        .map(|c| if c.is_ascii_whitespace() { '-' } else { c })
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    let s = if s.is_empty() {
        name.replace(['!', ':'], "")
            .chars()
            .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
            .collect()
    } else {
        s
    };
    s.chars().take(80).collect()
}

fn load_session(path: &Path) -> Option<MatrixSession> {
    let data = fs::read_to_string(path).ok()?;
    serde_json::from_str(&data).ok()
}

fn save_session(path: &Path, session: &MatrixSession) -> Result<()> {
    let data = serde_json::to_string(session)?;
    fs::write(path, data)?;
    Ok(())
}

fn load_device_id(path: &Path) -> Option<String> {
    let data = fs::read_to_string(path).ok()?;
    let id = data.trim().to_string();
    if id.is_empty() { None } else { Some(id) }
}

fn save_device_id(path: &Path, device_id: &str) -> Result<()> {
    fs::write(path, device_id)?;
    Ok(())
}

fn is_mentioned(
    event: &OriginalSyncRoomMessageEvent,
    bot_user_id: &OwnedUserId,
    body: &str,
) -> bool {
    // Primary: check m.mentions.user_ids
    if let Some(ref mentions) = event.content.mentions {
        if mentions.user_ids.contains(bot_user_id) {
            return true;
        }
    }
    // Fallback: check if body contains the bot's user ID string
    body.contains(bot_user_id.as_str())
}

pub async fn run(config: &Config, reset: bool) -> Result<()> {
    let mc = match &config.matrix {
        Some(mc) => mc,
        None => bail!("No [matrix] section in config.yaml"),
    };

    let url = Url::parse(&mc.homeserver)?;
    let host = url.host_str().unwrap_or("unknown").to_string();

    let output_dir = config.references_dir().join("matrix").join(&host);
    let store_dir = &mc.store_path;

    if reset {
        eprintln!("Resetting sync state…");
        // Preserve device_id, remove everything else
        let device_id = load_device_id(&store_dir.join("device_id"));
        if store_dir.exists() {
            fs::remove_dir_all(store_dir)?;
        }
        fs::create_dir_all(store_dir)?;
        if let Some(id) = &device_id {
            save_device_id(&store_dir.join("device_id"), id)?;
        }
    } else {
        fs::create_dir_all(store_dir)?;
    }

    let session_file = store_dir.join("session.json");
    let device_id_file = store_dir.join("device_id");

    let http_client = reqwest::Client::builder()
        .user_agent("lenny1")
        .min_tls_version(reqwest::tls::Version::TLS_1_2)
        .pool_idle_timeout(Duration::from_secs(300))
        .pool_max_idle_per_host(10)
        .tcp_keepalive(Duration::from_secs(60))
        .http2_keep_alive_interval(Duration::from_secs(30))
        .http2_keep_alive_timeout(Duration::from_secs(10))
        .build()?;

    let client = Client::builder()
        .homeserver_url(&mc.homeserver)
        .sqlite_store(store_dir, None)
        .http_client(http_client)
        .build()
        .await?;

    if !reset {
        if let Some(session) = load_session(&session_file) {
            eprintln!("Restoring saved session…");
            client.restore_session(session).await?;
        }
    }

    if client.session_meta().is_none() {
        eprintln!("Logging in as {}…", mc.username);
        let mut login = client
            .matrix_auth()
            .login_username(&mc.username, &mc.password)
            .initial_device_display_name("lenny1");

        if let Some(id) = load_device_id(&device_id_file) {
            login = login.device_id(&id);
        }

        login.send().await?;

        if let Some(session) = client.matrix_auth().session() {
            save_device_id(&device_id_file, session.meta.device_id.as_str())?;
            save_session(&session_file, &session)?;
        }
    }

    let rooms = client.joined_rooms();
    eprintln!(
        "Logged in as {} — {} joined room(s)",
        mc.username,
        rooms.len()
    );

    fs::create_dir_all(&output_dir)?;

    let bot_user_id = client.user_id().expect("should be logged in").to_owned();

    let out_dir = output_dir.clone();
    let config_clone = config.clone();
    let chat_lines: Arc<Mutex<HashMap<String, Vec<String>>>> = Arc::new(Mutex::new(HashMap::new()));

    client.add_event_handler(move |event: OriginalSyncRoomMessageEvent, room: Room| {
        let out_dir = out_dir.clone();
        let bot_user_id = bot_user_id.clone();
        let config = config_clone.clone();
        let chat_lines = chat_lines.clone();
        async move {
            let room_name = room.name().unwrap_or_else(|| room.room_id().to_string());
            let sender = event.sender.to_string();

            // Skip messages from ourselves
            if event.sender == bot_user_id {
                eprintln!("[{room_name}] (self) {sender}");
                return;
            }

            let body = match &event.content.msgtype {
                MessageType::Text(text) => text.body.clone(),
                _ => {
                    eprintln!("[{room_name}] {sender}: <non-text message>");
                    return;
                }
            };

            eprintln!("[{room_name}] {sender}: {body}");

            // Spawn all heavy work (file I/O, member lookup, LLM) off the sync loop
            let mentioned = is_mentioned(&event, &bot_user_id, &body);
            let event_sender = event.sender.clone();
            let event_id = event.event_id.to_string();
            let reply_to_event_id = event.event_id.clone();
            let thread_root = event.content.relates_to.as_ref().and_then(|rel| match rel {
                Relation::Thread(thread) => Some(thread.event_id.clone()),
                _ => None,
            });
            let timestamp: u64 = event.origin_server_ts.0.into();

            tokio::spawn(async move {
                let room_id = room.room_id().to_string();
                let sender_name = room
                    .get_member_no_sync(&event_sender)
                    .await
                    .ok()
                    .flatten()
                    .and_then(|m| m.display_name().map(|s| s.to_owned()));

                let line = json!({
                    "id": event_id,
                    "timestamp": timestamp,
                    "sender": sender,
                    "sender_name": sender_name,
                    "body": body,
                    "room": room_name,
                    "room_id": room_id,
                });

                let sanitized_name = sanitize_room_name(&room_name);
                let sanitized_id = sanitize_room_name(&room_id);
                let filename = if sanitized_name.is_empty() || sanitized_name == sanitized_id {
                    format!("{sanitized_id}.json")
                } else {
                    format!("{sanitized_id}-{sanitized_name}.json")
                };
                let path = out_dir.join(&filename);
                if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
                    let _ = writeln!(file, "{line}");
                }

                if !mentioned {
                    return;
                }

                // Skip messages older than 60 seconds
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_millis() as u64;
                let age_secs = now_ms.saturating_sub(timestamp) / 1000;
                if age_secs > 60 {
                    eprintln!("Skipping stale mention in {room_name} ({age_secs}s old): {body}");
                    return;
                }

                eprintln!("Mentioned in {room_name} by {sender}: {body}");

                match once::run_prompt(&config, &body).await {
                    Ok(result) if !result.skipped => {
                        let mut content = RoomMessageEventContent::text_plain(&result.answer);
                        if let Some(thread_root) = thread_root {
                            // Reply within the existing thread
                            let mut thread = Thread::plain(thread_root, reply_to_event_id);
                            thread.is_falling_back = false;
                            content.relates_to = Some(Relation::Thread(thread));
                        }
                        let t0 = std::time::Instant::now();
                        if let Err(e) = room.send(content).await {
                            eprintln!("Failed to send reply: {e}");
                            return;
                        }
                        eprintln!(
                            "Replied in {room_name} ({:?}): {}",
                            t0.elapsed(),
                            &result.answer
                        );

                        let session_id = format!("10-matrix-{sanitized_id}");
                        let ts = chrono::Utc::now().timestamp();
                        let user_line = serde_json::to_string(&json!({
                            "id": uuid::Uuid::new_v4().to_string(),
                            "timestamp": ts,
                            "sender": sender,
                            "body": body,
                        }))
                        .unwrap();
                        let bot_line = serde_json::to_string(&json!({
                            "id": uuid::Uuid::new_v4().to_string(),
                            "timestamp": chrono::Utc::now().timestamp(),
                            "sender": "lennybot",
                            "body": result.answer,
                        }))
                        .unwrap();

                        let mut lines_map = chat_lines.lock().unwrap();
                        let room_lines = lines_map.entry(session_id.clone()).or_default();
                        room_lines.push(user_line);
                        room_lines.push(bot_line);
                        let _ = cli_bot::save_chat_file(&config, &session_id, room_lines);
                    }
                    Ok(_) => {}
                    Err(e) => eprintln!("Agent error for {room_name}: {e}"),
                }
            });
        }
    });

    // Reaction handler: log emoji reactions to the same NDJSON files
    let out_dir2 = output_dir.clone();
    client.add_event_handler(move |event: OriginalSyncReactionEvent, room: Room| {
        let out_dir = out_dir2.clone();
        async move {
            let room_name = room.name().unwrap_or_else(|| room.room_id().to_string());
            let sender = event.sender.to_string();
            let emoji = event.content.relates_to.key.clone();
            let reacts_to = event.content.relates_to.event_id.to_string();

            eprintln!("[{room_name}] {sender} reacted {emoji} to {reacts_to}");

            let event_id = event.event_id.to_string();
            let timestamp: u64 = event.origin_server_ts.0.into();
            let room_id = room.room_id().to_string();

            tokio::spawn(async move {
                let line = json!({
                    "id": event_id,
                    "timestamp": timestamp,
                    "sender": sender,
                    "type": "reaction",
                    "body": emoji,
                    "reacts_to": reacts_to,
                    "room": room_name,
                    "room_id": room_id,
                });

                let sanitized_name = sanitize_room_name(&room_name);
                let sanitized_id = sanitize_room_name(&room_id);
                let filename = if sanitized_name.is_empty() || sanitized_name == sanitized_id {
                    format!("{sanitized_id}.json")
                } else {
                    format!("{sanitized_id}-{sanitized_name}.json")
                };
                let path = out_dir.join(&filename);
                if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(&path) {
                    let _ = writeln!(file, "{line}");
                }
            });
        }
    });

    // Auto-join rooms we're invited to
    let bot_user_id_for_invite = client.user_id().expect("should be logged in").to_owned();
    client.add_event_handler(move |event: StrippedRoomMemberEvent, room: Room| {
        let bot_user_id = bot_user_id_for_invite.clone();
        async move {
            // Only act on invites addressed to us
            if event.state_key != bot_user_id {
                return;
            }
            let room_name = room.room_id().to_string();
            eprintln!("Invited to {room_name} by {}", event.sender);
            tokio::spawn(async move {
                // Retry a few times — the room may not be ready immediately
                for attempt in 0..3 {
                    match room.join().await {
                        Ok(_) => {
                            eprintln!("Joined {room_name}");
                            return;
                        }
                        Err(e) => {
                            eprintln!("Failed to join {room_name} (attempt {}): {e}", attempt + 1);
                            tokio::time::sleep(Duration::from_secs(2)).await;
                        }
                    }
                }
            });
        }
    });

    eprintln!("Syncing (sliding sync)…");

    let sliding_sync = client
        .sliding_sync("lenny")?
        .version(Version::Native)
        .add_list(
            SlidingSyncList::builder("all")
                .sync_mode(SlidingSyncMode::new_growing(50))
                .timeline_limit(10),
        )
        .build()
        .await?;

    let mut stream = std::pin::pin!(sliding_sync.sync());

    while let Some(result) = stream.next().await {
        match result {
            Ok(update) => {
                if !update.rooms.is_empty() {
                    eprintln!("Sync: {} room(s) updated", update.rooms.len());
                }
            }
            Err(e) => {
                eprintln!("Sliding sync error: {e}");
                break;
            }
        }
    }

    Ok(())
}
