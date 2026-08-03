use std::io::{self, Stdout};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use crossterm::event::{self, Event, KeyCode};
use crossterm::execute;
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Tabs};
use ratatui::Terminal;
use reqwest::Client;
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Deserialize, Default, Clone)]
struct HealthPayload {
    status: Option<String>,
    active_model: Option<String>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct SummaryPayload {
    total_results: Option<u64>,
    statuses: Option<Value>,
    roles: Option<Value>,
    latest: Option<Vec<Value>>,
}

#[derive(Debug, Deserialize, Default, Clone)]
struct ChatPayload {
    status: Option<String>,
    chat_mode: Option<String>,
    payload: Option<Value>,
}

#[derive(Debug, Clone)]
struct App {
    api_base: String,
    selected_tab: usize,
    input: String,
    status_line: String,
    health: HealthPayload,
    summary: SummaryPayload,
    last_chat: Option<ChatPayload>,
    last_refresh: Option<Instant>,
}

impl App {
    fn new(api_base: String) -> Self {
        Self {
            api_base,
            selected_tab: 0,
            input: String::new(),
            status_line: "Ready".to_string(),
            health: HealthPayload::default(),
            summary: SummaryPayload::default(),
            last_chat: None,
            last_refresh: None,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let api_base = std::env::var("MX2LM_API_BASE").unwrap_or_else(|_| "http://127.0.0.1:8000".to_string());
    let client = Client::builder().build().context("failed to create http client")?;

    enable_raw_mode().context("failed to enable raw mode")?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen).context("failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend).context("failed to create terminal")?;

    let mut app = App::new(api_base);
    refresh(&client, &mut app).await.ok();

    let result = run(&client, &mut terminal, &mut app).await;

    disable_raw_mode().ok();
    execute!(terminal.backend_mut(), LeaveAlternateScreen).ok();
    terminal.show_cursor().ok();

    result
}

async fn run(client: &Client, terminal: &mut Terminal<CrosstermBackend<Stdout>>, app: &mut App) -> Result<()> {
    loop {
        terminal.draw(|frame| render(frame, app)).context("failed to draw UI")?;

        if event::poll(Duration::from_millis(250)).context("event poll failed")? {
            if let Event::Key(key) = event::read().context("event read failed")? {
                match key.code {
                    KeyCode::Char('q') => break,
                    KeyCode::Char('1') => app.selected_tab = 0,
                    KeyCode::Char('2') => app.selected_tab = 1,
                    KeyCode::Char('3') => app.selected_tab = 2,
                    KeyCode::Char('r') => {
                        refresh(client, app).await.ok();
                    }
                    KeyCode::Enter => {
                        if !app.input.trim().is_empty() {
                            submit_chat(client, app).await.ok();
                        }
                    }
                    KeyCode::Backspace => {
                        app.input.pop();
                    }
                    KeyCode::Char(c) => {
                        app.input.push(c);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}

async fn refresh(client: &Client, app: &mut App) -> Result<()> {
    let health = client
        .get(format!("{}/health", app.api_base))
        .send()
        .await
        .context("health request failed")?
        .json::<HealthPayload>()
        .await
        .context("health decode failed")?;

    let status = client
        .get(format!("{}/micronaut/status", app.api_base))
        .send()
        .await
        .context("summary request failed")?
        .json::<Value>()
        .await
        .context("summary decode failed")?;

    let summary = serde_json::from_value(status.get("summary").cloned().unwrap_or(Value::Null)).unwrap_or_default();

    app.health = health;
    app.summary = summary;
    app.last_refresh = Some(Instant::now());
    app.status_line = "Refreshed from Micronaut API".to_string();
    Ok(())
}

async fn submit_chat(client: &Client, app: &mut App) -> Result<()> {
    let run = !app.input.trim_start().starts_with("/plan ");
    let payload = serde_json::json!({
        "message": app.input,
        "run": run,
    });
    let chat = client
        .post(format!("{}/chat", app.api_base))
        .json(&payload)
        .send()
        .await
        .context("chat request failed")?
        .json::<ChatPayload>()
        .await
        .context("chat decode failed")?;

    let mode = chat.chat_mode.clone().unwrap_or_else(|| "unknown".to_string());
    app.last_chat = Some(chat);
    app.status_line = format!("Submitted chat request via mode: {}", mode);
    app.input.clear();
    refresh(client, app).await.ok();
    Ok(())
}

fn render(frame: &mut ratatui::Frame, app: &App) {
    let areas = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(8),
            Constraint::Length(3),
            Constraint::Length(3),
        ])
        .split(frame.size());

    let title = Paragraph::new("MX2LM Micronaut Orchestrator")
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD))
        .block(Block::default().title("Rust TUI").borders(Borders::ALL));
    frame.render_widget(title, areas[0]);

    let tabs = Tabs::new(vec!["Chat", "Status", "Summary"])
        .select(app.selected_tab)
        .block(Block::default().title("Views").borders(Borders::ALL))
        .highlight_style(Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD));
    frame.render_widget(tabs, areas[1]);

    match app.selected_tab {
        0 => render_chat(frame, areas[2], app),
        1 => render_status(frame, areas[2], app),
        _ => render_summary(frame, areas[2], app),
    }

    let input = Paragraph::new(app.input.as_str())
        .block(Block::default().title("Input").borders(Borders::ALL))
        .style(Style::default().fg(Color::White));
    frame.render_widget(input, areas[3]);

    let status = Paragraph::new(app.status_line.as_str())
        .block(Block::default().title("Status").borders(Borders::ALL))
        .style(Style::default().fg(Color::Green));
    frame.render_widget(status, areas[4]);
}

fn render_chat(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    let mut lines = vec![
        Line::from("Type a task and press Enter."),
        Line::from("Use /status, /summary, /workers, /resume <id>, /rerun <id>, /plan <task>, /swarm <task>, /generate <task>, /code <task>."),
        Line::from(""),
    ];
    if let Some(chat) = &app.last_chat {
        lines.push(Line::from(format!("Status: {}", chat.status.clone().unwrap_or_default())));
        lines.push(Line::from(format!("Mode: {}", chat.chat_mode.clone().unwrap_or_default())));
        if let Some(payload) = &chat.payload {
            let pretty = serde_json::to_string_pretty(payload).unwrap_or_else(|_| "{}".to_string());
            for line in pretty.lines().take(20) {
                lines.push(Line::from(line.to_string()));
            }
        }
    }
    let para = Paragraph::new(lines).block(Block::default().title("Micronaut Chat").borders(Borders::ALL));
    frame.render_widget(para, area);
}

fn render_status(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    let items = vec![
        ListItem::new(Line::from(vec![Span::styled("Server Status: ", Style::default().fg(Color::Cyan)), Span::raw(app.health.status.clone().unwrap_or_else(|| "unknown".to_string()))])),
        ListItem::new(Line::from(vec![Span::styled("Active Model: ", Style::default().fg(Color::Cyan)), Span::raw(app.health.active_model.clone().unwrap_or_else(|| "(none)".to_string()))])),
        ListItem::new(Line::from(format!("API Base: {}", app.api_base))),
        ListItem::new(Line::from(format!("Last Refresh: {}", app.last_refresh.map(|_| "recent".to_string()).unwrap_or_else(|| "never".to_string())))),
        ListItem::new(Line::from("Keys: 1-3 tabs, r refresh, Enter submit, q quit")),
    ];
    let list = List::new(items).block(Block::default().title("Micronaut Status").borders(Borders::ALL));
    frame.render_widget(list, area);
}

fn render_summary(frame: &mut ratatui::Frame, area: ratatui::layout::Rect, app: &App) {
    let mut items = vec![
        ListItem::new(format!("Total Results: {}", app.summary.total_results.unwrap_or(0))),
        ListItem::new(format!("Statuses: {}", app.summary.statuses.clone().unwrap_or(Value::Null))),
        ListItem::new(format!("Roles: {}", app.summary.roles.clone().unwrap_or(Value::Null))),
    ];
    if let Some(latest) = &app.summary.latest {
        for item in latest.iter().take(5) {
            items.push(ListItem::new(format!("Latest: {}", item)));
        }
    }
    let list = List::new(items).block(Block::default().title("Micronaut Summary").borders(Borders::ALL));
    frame.render_widget(list, area);
}
