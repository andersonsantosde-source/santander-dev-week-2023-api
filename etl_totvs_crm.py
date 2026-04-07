"""
╔══════════════════════════════════════════════════════════════════╗
║         TOTVS CRM — Pipeline ETL com IA Generativa              ║
║   Extração → Transformação (Claude AI) → Carregamento           ║
╚══════════════════════════════════════════════════════════════════╝

Objetivo:
  Ler base de clientes CRM, gerar sugestões personalizadas de
  atendimento para agregar valor aos produtos Totvs via Claude API,
  e salvar os resultados enriquecidos em JSON e CSV.

Instalação:
  pip install pandas anthropic rich
"""

# ──────────────────────────────────────────────────────────────────
# CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────────────
import os
import json
import pandas as pd
import anthropic
from datetime import datetime, date
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel
from rich import print as rprint

console = Console()

# Substitua pela sua chave ou defina a variável de ambiente
# ANTHROPIC_API_KEY = "sk-ant-..."
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "SUA_CHAVE_AQUI")

CSV_INPUT  = "crm_totvs.csv"
JSON_OUTPUT = "crm_totvs_enriquecido.json"
CSV_OUTPUT  = "crm_totvs_sugestoes.csv"


# ──────────────────────────────────────────────────────────────────
# ETAPA 1 — EXTRACT
# ──────────────────────────────────────────────────────────────────
def extract(csv_path: str) -> list[dict]:
    """Lê o CSV de clientes e retorna lista de dicionários estruturados."""
    console.rule("[bold cyan]📥 EXTRACT — Lendo base CRM[/bold cyan]")

    df = pd.read_csv(csv_path)

    # Normaliza colunas
    df.columns = df.columns.str.strip()

    # Calcula dias desde última interação
    df["Ultima_Interacao"] = pd.to_datetime(df["Ultima_Interacao"])
    df["Dias_Sem_Contato"] = (pd.Timestamp.now() - df["Ultima_Interacao"]).dt.days

    # Classifica risco de churn
    def risco_churn(row):
        score = 0
        if row["NPS"] <= 5:       score += 3
        elif row["NPS"] <= 7:     score += 1
        if row["Tickets_Abertos"] >= 5: score += 2
        if row["Dias_Sem_Contato"] > 90: score += 2
        if score >= 5: return "🔴 Alto"
        if score >= 2: return "🟡 Médio"
        return "🟢 Baixo"

    df["Risco_Churn"] = df.apply(risco_churn, axis=1)

    # Adiciona campo para as sugestões (será preenchido na etapa Transform)
    df["Sugestoes_IA"] = None

    clients = df.to_dict(orient="records")

    # Exibe resumo no terminal
    table = Table(title="Clientes Carregados", show_lines=True)
    table.add_column("ID",    style="dim")
    table.add_column("Nome",  style="bold white")
    table.add_column("Empresa")
    table.add_column("Produto",       style="cyan")
    table.add_column("NPS",           style="bold")
    table.add_column("Tickets",       justify="center")
    table.add_column("Risco Churn",   justify="center")

    for c in clients:
        nps_color = "green" if c["NPS"] >= 8 else ("yellow" if c["NPS"] >= 6 else "red")
        table.add_row(
            str(c["ClientID"]),
            c["Nome"],
            c["Empresa"],
            c["Produto_Atual"],
            f"[{nps_color}]{c['NPS']}[/{nps_color}]",
            str(c["Tickets_Abertos"]),
            c["Risco_Churn"],
        )

    console.print(table)
    console.print(f"\n✅ [green]{len(clients)} clientes extraídos com sucesso.[/green]\n")
    return clients


# ──────────────────────────────────────────────────────────────────
# ETAPA 2 — TRANSFORM
# ──────────────────────────────────────────────────────────────────

PROMPT_SISTEMA = """
Você é um especialista sênior em Customer Success e CRM da TOTVS, maior empresa
de tecnologia para gestão empresarial do Brasil. Seu papel é analisar o perfil de
cada cliente e gerar sugestões práticas, personalizadas e estratégicas de
atendimento para aumentar a satisfação, reduzir churn e identificar oportunidades
de upsell/cross-sell nos produtos Totvs.

Produtos do portfólio relevantes:
- Protheus ERP: gestão integrada (financeiro, RH, estoque, fiscal)
- TOTVS Varejo Omnichannel: PDV, e-commerce, gestão de lojas
- TOTVS Saúde: prontuário eletrônico, faturamento hospitalar
- Totvs Agro: gestão de fazendas, colheita, compliance rural
- TOTVS Educacional: gestão escolar, secretaria, financeiro
- Totvs Jurídico: contratos, processos, compliance
- TOTVS Analytics: BI e dashboards integrados
- TOTVS RH: folha, benefícios, gestão de talentos

Responda SEMPRE em JSON válido com esta estrutura exata:
{
  "resumo_perfil": "string — análise breve do momento do cliente (2 frases)",
  "acoes_imediatas": ["ação 1", "ação 2", "ação 3"],
  "oportunidades_upsell": ["oportunidade 1", "oportunidade 2"],
  "mensagem_personalizada": "string — mensagem direta ao cliente (máx 120 chars)",
  "prioridade_contato": "URGENTE | ALTA | MÉDIA | BAIXA"
}
"""

def build_prompt(client: dict) -> str:
    return f"""
Analise este cliente e gere as sugestões de atendimento:

- Nome: {client['Nome']}
- Empresa: {client['Empresa']} ({client['Segmento']}, {client['Funcionarios']} funcionários)
- Estado: {client['Estado']}
- Produto atual: {client['Produto_Atual']} (cliente há {client['Tempo_Cliente_Anos']} ano(s))
- NPS: {client['NPS']}/10
- Tickets em aberto: {client['Tickets_Abertos']}
- Dias sem contato: {client['Dias_Sem_Contato']}
- Faturamento anual: R$ {client['Faturamento_Anual_R$']:,}
- Risco de churn estimado: {client['Risco_Churn']}
"""


def generate_suggestions(client: dict, ai_client: anthropic.Anthropic) -> dict:
    """Chama a API Claude para gerar sugestões personalizadas."""
    message = ai_client.messages.create(
        model="claude-opus-4-5",
        max_tokens=600,
        system=PROMPT_SISTEMA,
        messages=[{"role": "user", "content": build_prompt(client)}],
    )

    raw = message.content[0].text.strip()

    # Remove eventuais blocos de código markdown
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def transform(clients: list[dict]) -> list[dict]:
    """Enriquece cada cliente com sugestões geradas por IA."""
    console.rule("[bold magenta]🔄 TRANSFORM — Gerando sugestões com Claude AI[/bold magenta]")

    if ANTHROPIC_API_KEY == "SUA_CHAVE_AQUI":
        console.print("[bold red]⚠️  ANTHROPIC_API_KEY não configurada! Usando sugestões de demonstração.[/bold red]\n")
        return _transform_mock(clients)

    ai_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    for client in track(clients, description="Processando clientes..."):
        try:
            suggestions = generate_suggestions(client, ai_client)
            client["Sugestoes_IA"] = suggestions
            prioridade = suggestions.get("prioridade_contato", "MÉDIA")
            cor = {"URGENTE": "red", "ALTA": "yellow", "MÉDIA": "cyan", "BAIXA": "green"}.get(prioridade, "white")
            console.print(
                f"  ✓ [bold]{client['Nome']}[/bold] — "
                f"[{cor}]{prioridade}[/{cor}] | "
                f"{suggestions['mensagem_personalizada'][:60]}..."
            )
        except Exception as e:
            console.print(f"  ✗ [red]Erro em {client['Nome']}: {e}[/red]")
            client["Sugestoes_IA"] = {"erro": str(e)}

    console.print(f"\n✅ [green]Transformação concluída para {len(clients)} clientes.[/green]\n")
    return clients


def _transform_mock(clients: list[dict]) -> list[dict]:
    """Sugestões de demonstração quando a API key não está configurada."""
    mocks = [
        {
            "resumo_perfil": "Cliente engajado com alto volume de tickets pendentes, indicando necessidade de suporte técnico especializado.",
            "acoes_imediatas": [
                "Agendar call de saúde técnica em até 48h",
                "Alocar consultor especialista no produto",
                "Oferecer pacote de suporte premium"
            ],
            "oportunidades_upsell": [
                "TOTVS Analytics para dashboards operacionais",
                "TOTVS RH integrado ao ERP atual"
            ],
            "mensagem_personalizada": "Olá! Identificamos pontos de melhoria e temos soluções para elevar sua operação.",
            "prioridade_contato": "ALTA"
        },
        {
            "resumo_perfil": "Cliente promotor com excelente NPS, ideal para programa de cases de sucesso e expansão de módulos.",
            "acoes_imediatas": [
                "Convidar para programa Totvs Champion",
                "Apresentar roadmap de novidades do produto",
                "Propor expansão para filiais/unidades"
            ],
            "oportunidades_upsell": [
                "TOTVS Analytics Premium",
                "Integração com marketplace digital"
            ],
            "mensagem_personalizada": "Você é um parceiro exemplar! Temos novidades exclusivas para clientes como você.",
            "prioridade_contato": "MÉDIA"
        },
    ]

    for i, client in enumerate(clients):
        client["Sugestoes_IA"] = mocks[i % len(mocks)]

    console.print(f"\n✅ [yellow]Sugestões de demonstração aplicadas para {len(clients)} clientes.[/yellow]\n")
    return clients


# ──────────────────────────────────────────────────────────────────
# ETAPA 3 — LOAD
# ──────────────────────────────────────────────────────────────────
def load(clients: list[dict], json_path: str, csv_path: str) -> None:
    """Salva os dados enriquecidos em JSON completo e CSV resumido."""
    console.rule("[bold green]📤 LOAD — Salvando resultados[/bold green]")

    # ── JSON completo ──────────────────────────────────────────────
    def serialize(obj):
        if isinstance(obj, (datetime, date, pd.Timestamp)):
            return obj.isoformat()
        raise TypeError(f"Tipo não serializável: {type(obj)}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"gerado_em": datetime.now().isoformat(), "clientes": clients},
                  f, ensure_ascii=False, indent=2, default=serialize)
    console.print(f"  💾 JSON completo salvo → [bold]{json_path}[/bold]")

    # ── CSV resumido ───────────────────────────────────────────────
    rows = []
    for c in clients:
        s = c.get("Sugestoes_IA") or {}
        acoes = " | ".join(s.get("acoes_imediatas", []))
        upsell = " | ".join(s.get("oportunidades_upsell", []))
        rows.append({
            "ClientID":             c["ClientID"],
            "Nome":                 c["Nome"],
            "Empresa":              c["Empresa"],
            "Segmento":             c["Segmento"],
            "Produto_Atual":        c["Produto_Atual"],
            "NPS":                  c["NPS"],
            "Tickets_Abertos":      c["Tickets_Abertos"],
            "Risco_Churn":          c["Risco_Churn"],
            "Prioridade_Contato":   s.get("prioridade_contato", "—"),
            "Resumo_Perfil":        s.get("resumo_perfil", "—"),
            "Acoes_Imediatas":      acoes,
            "Oportunidades_Upsell": upsell,
            "Mensagem_Cliente":     s.get("mensagem_personalizada", "—"),
        })

    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
    console.print(f"  📊 CSV resumido salvo  → [bold]{csv_path}[/bold]")

    # ── Sumário executivo ──────────────────────────────────────────
    urgentes = sum(1 for c in clients if (c.get("Sugestoes_IA") or {}).get("prioridade_contato") == "URGENTE")
    altas    = sum(1 for c in clients if (c.get("Sugestoes_IA") or {}).get("prioridade_contato") == "ALTA")

    console.print(Panel(
        f"[bold]📋 Sumário Executivo[/bold]\n\n"
        f"  Clientes processados : [cyan]{len(clients)}[/cyan]\n"
        f"  Contato URGENTE      : [red]{urgentes}[/red] clientes\n"
        f"  Prioridade ALTA      : [yellow]{altas}[/yellow] clientes\n"
        f"  Arquivos gerados     : [green]{json_path}[/green] · [green]{csv_path}[/green]",
        title="TOTVS CRM — ETL Concluído",
        border_style="green",
    ))


# ──────────────────────────────────────────────────────────────────
# EXECUÇÃO PRINCIPAL
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold cyan]TOTVS CRM — Pipeline ETL com IA Generativa[/bold cyan]\n"
        "Extração · Transformação via Claude · Carregamento",
        border_style="cyan",
    ))

    # 1. Extract
    clients = extract(CSV_INPUT)

    # 2. Transform
    clients = transform(clients)

    # 3. Load
    load(clients, JSON_OUTPUT, CSV_OUTPUT)
