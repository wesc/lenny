mod agent;
mod config;
mod datasets;
mod runner;
mod tools;

use anyhow::Result;
use clap::{Parser, builder::PossibleValuesParser};
use datasets::{CheckMethod, DatasetKind};
use openrouter_rs::OpenRouterClient;
use std::path::PathBuf;
use std::time::Instant;

fn dataset_value_parser() -> PossibleValuesParser {
    let mut names: Vec<&str> = datasets::dataset_names();
    names.push("all");
    PossibleValuesParser::new(names)
}

#[derive(Parser)]
#[command(version, about = "Eval harness for reasoning loop + response model")]
struct Cli {
    /// Dataset to run, or "all" to run every dataset
    #[arg(value_parser = dataset_value_parser())]
    dataset: String,

    /// Path to config file (reads api_key, reasoning_model, response_model)
    #[arg(short, long, default_value = "../lenny1/config.yaml")]
    config: PathBuf,

    /// Override reasoning model
    #[arg(long)]
    reasoning_model: Option<String>,

    /// Override response model
    #[arg(long)]
    response_model: Option<String>,

    /// Override judge model (for LLM-as-judge evals)
    #[arg(long)]
    judge_model: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    let mut config = config::Config::load(&cli.config)?;
    if let Some(m) = cli.reasoning_model {
        config.reasoning_model = m;
    }
    if let Some(m) = cli.response_model {
        config.response_model = m;
    }
    if let Some(m) = cli.judge_model {
        config.judge_model = m;
    }

    let client = OpenRouterClient::builder()
        .api_key(&config.api_key)
        .build()?;

    eprintln!(
        "reasoning: {}  response: {}  judge: {}",
        config.reasoning_model, config.response_model, config.judge_model
    );

    config.response_effort = agent::probe_min_effort(&client, &config.response_model).await?;
    eprintln!();

    let all = datasets::all_datasets();
    let datasets: Vec<_> = if cli.dataset == "all" {
        all
    } else {
        all.into_iter().filter(|d| d.name == cli.dataset).collect()
    };

    let mut total_passed = 0;
    let mut total_count = 0;

    for ds in &datasets {
        match &ds.kind {
            DatasetKind::ToolUse { tools, evals } => {
                eprintln!("=== {} ({} evals) ===", ds.name, evals.len());

                let mut passed = 0;
                let ds_start = Instant::now();

                for eval in evals {
                    eprint!("  {} ... ", eval.name);
                    let eval_start = Instant::now();

                    match runner::run_eval(&client, &config, ds.system, eval.prompt, tools).await {
                        Ok(result) => {
                            let elapsed = eval_start.elapsed().as_secs_f64();
                            let (pass, reason) = (eval.check)(&result);
                            if pass {
                                passed += 1;
                                eprintln!(
                                    "PASS ({:.1}s, {} tool calls)",
                                    elapsed,
                                    result.tool_events.len()
                                );
                            } else {
                                eprintln!("FAIL ({elapsed:.1}s)");
                                eprintln!("    reason: {reason}");
                                eprintln!("    answer: {}", result.answer);
                                for event in &result.tool_events {
                                    eprintln!(
                                        "    [{}] {} -> {}",
                                        event.tool, event.args, event.result
                                    );
                                }
                            }
                        }
                        Err(e) => {
                            let elapsed = eval_start.elapsed().as_secs_f64();
                            eprintln!("ERROR ({elapsed:.1}s): {e}");
                        }
                    }
                }

                let ds_elapsed = ds_start.elapsed().as_secs_f64();
                eprintln!("\n  {passed}/{} passed in {ds_elapsed:.1}s\n", evals.len());
                total_passed += passed;
                total_count += evals.len();
            }
            DatasetKind::ResponseOnly { evals } => {
                eprintln!("=== {} ({} evals) ===", ds.name, evals.len());

                let mut passed = 0;
                let ds_start = Instant::now();

                for eval in evals {
                    eprint!("  {} ... ", eval.name);
                    let eval_start = Instant::now();

                    match runner::run_response_only(
                        &client,
                        &config,
                        ds.system,
                        eval.prompt,
                        eval.tool_context,
                    )
                    .await
                    {
                        Ok(result) => {
                            let (pass, reason) = match &eval.check {
                                CheckMethod::Judge(criteria) => {
                                    match runner::judge(
                                        &client,
                                        &config.judge_model,
                                        &result.answer,
                                        criteria,
                                    )
                                    .await
                                    {
                                        Ok(j) => (j.pass, j.reason),
                                        Err(e) => (false, format!("judge error: {e}")),
                                    }
                                }
                                CheckMethod::Exact(check_fn) => check_fn(&result),
                            };
                            let elapsed = eval_start.elapsed().as_secs_f64();
                            if pass {
                                passed += 1;
                                eprintln!("PASS ({elapsed:.1}s)");
                            } else {
                                eprintln!("FAIL ({elapsed:.1}s)");
                                if let CheckMethod::Judge(criteria) = &eval.check {
                                    eprintln!("    criteria: {criteria}");
                                }
                                eprintln!("    reason: {reason}");
                                eprintln!("    answer: {}", result.answer);
                            }
                        }
                        Err(e) => {
                            let elapsed = eval_start.elapsed().as_secs_f64();
                            eprintln!("ERROR ({elapsed:.1}s): {e}");
                        }
                    }
                }

                let ds_elapsed = ds_start.elapsed().as_secs_f64();
                eprintln!("\n  {passed}/{} passed in {ds_elapsed:.1}s\n", evals.len());
                total_passed += passed;
                total_count += evals.len();
            }
        }
    }

    if datasets.len() > 1 {
        eprintln!("Total: {total_passed}/{total_count}");
    }

    if total_passed < total_count {
        std::process::exit(1);
    }

    Ok(())
}
