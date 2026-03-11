use anyhow::Result;
use rusqlite::Connection;
use std::path::Path;
use std::sync::Once;

use crate::embed;

const EMBEDDING_DIM: usize = 384;

/// Register the sqlite-vec extension (idempotent).
fn ensure_vec_extension() {
    static INIT: Once = Once::new();
    INIT.call_once(|| unsafe {
        rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute::<
            *const (),
            unsafe extern "C" fn(
                *mut rusqlite::ffi::sqlite3,
                *mut *mut i8,
                *const rusqlite::ffi::sqlite3_api_routines,
            ) -> i32,
        >(
            sqlite_vec::sqlite3_vec_init as *const ()
        )));
    });
}

/// Open (or create) the lenny2 database with sqlite-vec loaded.
pub fn open(path: &Path) -> Result<Connection> {
    ensure_vec_extension();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(path)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    init_schema(&conn)?;
    Ok(conn)
}

fn init_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rumination_id TEXT NOT NULL,
            cluster_index INTEGER NOT NULL,
            summary TEXT NOT NULL,
            page_paths TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_clusters_rumination
            ON clusters(rumination_id);
        ",
    )?;

    // Create the virtual vec table for centroids.
    // sqlite-vec uses a separate virtual table linked by rowid.
    conn.execute_batch(&format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS cluster_embeddings USING vec0(
            embedding float[{EMBEDDING_DIM}]
        );"
    ))?;

    Ok(())
}

/// Insert a cluster with its centroid embedding. Returns the row id.
pub fn insert_cluster(
    conn: &Connection,
    rumination_id: &str,
    cluster_index: usize,
    summary: &str,
    page_paths: &[String],
    centroid: &[f32],
) -> Result<i64> {
    let pages_json = serde_json::to_string(page_paths)?;

    conn.execute(
        "INSERT INTO clusters (rumination_id, cluster_index, summary, page_paths) VALUES (?1, ?2, ?3, ?4)",
        rusqlite::params![rumination_id, cluster_index as i64, summary, pages_json],
    )?;
    let rowid = conn.last_insert_rowid();

    conn.execute(
        "INSERT INTO cluster_embeddings (rowid, embedding) VALUES (?1, ?2)",
        rusqlite::params![rowid, embed::f32_to_bytes(centroid)],
    )?;

    Ok(rowid)
}

/// Find the top-n closest cluster summaries to a query embedding.
pub fn find_nearest_clusters(
    conn: &Connection,
    query_embedding: &[f32],
    n: usize,
) -> Result<Vec<ClusterResult>> {
    let mut stmt = conn.prepare(
        "SELECT
            c.id, c.rumination_id, c.cluster_index, c.summary, c.page_paths,
            ce.distance
        FROM cluster_embeddings ce
        JOIN clusters c ON c.id = ce.rowid
        WHERE ce.embedding MATCH ?1
            AND k = ?2
        ORDER BY ce.distance",
    )?;

    let rows = stmt.query_map(
        rusqlite::params![embed::f32_to_bytes(query_embedding), n as i64],
        |row| {
            Ok(ClusterResult {
                id: row.get(0)?,
                rumination_id: row.get(1)?,
                cluster_index: row.get(2)?,
                summary: row.get(3)?,
                page_paths: row.get(4)?,
                distance: row.get(5)?,
            })
        },
    )?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}

/// Delete all clusters for a given rumination_id (for re-clustering).
pub fn delete_clusters_for_rumination(conn: &Connection, rumination_id: &str) -> Result<()> {
    // Get rowids to delete from vec table
    let mut stmt = conn.prepare("SELECT id FROM clusters WHERE rumination_id = ?1")?;
    let ids: Vec<i64> = stmt
        .query_map([rumination_id], |row| row.get(0))?
        .collect::<Result<Vec<_>, _>>()?;

    for id in &ids {
        conn.execute("DELETE FROM cluster_embeddings WHERE rowid = ?1", [id])?;
    }

    conn.execute(
        "DELETE FROM clusters WHERE rumination_id = ?1",
        [rumination_id],
    )?;
    Ok(())
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct ClusterResult {
    pub id: i64,
    pub rumination_id: String,
    pub cluster_index: i64,
    pub summary: String,
    pub page_paths: String,
    pub distance: f64,
}
