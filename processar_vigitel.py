"""
processar_vigitel.py
--------------------
Lê os arquivos brutos do VIGITEL, aplica os mapeamentos do dicionário,
calcula prevalências com peso (pesorake) e gera dados.json para o dashboard.

Como usar:
    1. Coloque os arquivos .xls/.xlsx do VIGITEL na pasta 'dados/'
    2. Execute:  python processar_vigitel.py
    3. O arquivo dados.json será gerado na mesma pasta do script
    4. Abra index.html no navegador (ou publique no GitHub Pages)
"""

import os
import glob
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuração ─────────────────────────────────────────────────────────────

PASTA_DADOS  = "dados"          # pasta com os arquivos VIGITEL brutos
SAIDA_JSON   = "dados.json"     # arquivo de saída

# Colunas que vamos usar
COLUNAS = [
    "ano", "cidade",
    "q69",          # cor/raça (1=branca,2=preta,3=amarela,4=parda,5=indígena)
    "q7",           # sexo (1=masculino, 2=feminino)
    "fesc",         # escolaridade (1=0-8 anos, 2=9-11 anos, 3=12+ anos)
    "pesorake",     # peso amostral
    # Indicadores alimentares
    "hortareg", "frutareg", "flvreg", "refritl5",
    # Outros indicadores
    "obesid", "excpeso", "diab", "hart", "inativo", "fumante", "imc",
]

# ── Mapeamentos ───────────────────────────────────────────────────────────────

MAPA_RACA = {
    1: "Branca",
    2: "Preta",
    3: "Amarela",
    4: "Parda",
    5: "Indígena",
    80: "Outra",
}

MAPA_SEXO = {1: "Masculino", 2: "Feminino"}

MAPA_ESC = {
    1: "0 a 8 anos",
    2: "9 a 11 anos",
    3: "12 anos ou mais",
}

MAPA_CIDADE = {
    1: "Aracaju", 2: "Belém", 3: "Belo Horizonte", 4: "Boa Vista",
    5: "Campo Grande", 6: "Cuiabá", 7: "Curitiba", 8: "Florianópolis",
    9: "Fortaleza", 10: "Goiânia", 11: "João Pessoa", 12: "Macapá",
    13: "Maceió", 14: "Manaus", 15: "Natal", 16: "Palmas",
    17: "Porto Alegre", 18: "Porto Velho", 19: "Recife", 20: "Rio Branco",
    21: "Rio de Janeiro", 22: "Salvador", 23: "São Luís", 24: "São Paulo",
    25: "Teresina", 26: "Vitória", 27: "Porto Alegre",
}

MAPA_REGIAO = {
    "Aracaju": "Nordeste", "Belém": "Norte", "Belo Horizonte": "Sudeste",
    "Boa Vista": "Norte", "Campo Grande": "Centro-Oeste", "Cuiabá": "Centro-Oeste",
    "Curitiba": "Sul", "Florianópolis": "Sul", "Fortaleza": "Nordeste",
    "Goiânia": "Centro-Oeste", "João Pessoa": "Nordeste", "Macapá": "Norte",
    "Maceió": "Nordeste", "Manaus": "Norte", "Natal": "Nordeste",
    "Palmas": "Norte", "Porto Alegre": "Sul", "Porto Velho": "Norte",
    "Recife": "Nordeste", "Rio Branco": "Norte", "Rio de Janeiro": "Sudeste",
    "Salvador": "Nordeste", "São Luís": "Nordeste", "São Paulo": "Sudeste",
    "Teresina": "Nordeste", "Vitória": "Sudeste",
}

# Raças consideradas "negra" (preta + parda)
RACA_NEGRA = {"Preta", "Parda"}

INDICADORES = {
    "hortareg":  "Hortaliças ≥5x/sem",
    "frutareg":  "Frutas ≥5x/sem",
    "flvreg":    "Frutas e hortaliças ≥5x/sem",
    "refritl5":  "Refrigerante ≥5x/sem",
    "obesid":    "Obesidade",
    "excpeso":   "Excesso de peso",
    "diab":      "Diabetes",
    "hart":      "Hipertensão",
    "inativo":   "Inatividade física",
    "fumante":   "Tabagismo",
}

# ── Funções auxiliares ────────────────────────────────────────────────────────

def prevalencia_ponderada(df, indicador, peso="pesorake"):
    """Calcula prevalência ponderada pelo peso amostral."""
    sub = df[[indicador, peso]].dropna()
    sub = sub[sub[indicador].isin([0, 1])]
    if len(sub) == 0:
        return None, 0
    prev = (sub[indicador] * sub[peso]).sum() / sub[peso].sum() * 100
    return round(float(prev), 1), len(sub)


def calcular_cruzamento(df, indicador, grupos, col_grupo, peso="pesorake"):
    """Retorna lista de dicts com prevalência por grupo."""
    resultado = []
    for grupo in sorted(df[col_grupo].dropna().unique()):
        sub = df[df[col_grupo] == grupo]
        prev, n = prevalencia_ponderada(sub, indicador, peso)
        if prev is not None:
            resultado.append({
                "grupo": str(grupo),
                "prevalencia": prev,
                "n": int(n),
            })
    return resultado


# ── Leitura dos arquivos ──────────────────────────────────────────────────────

def ler_arquivo(caminho):
    ext = os.path.splitext(caminho)[1].lower()
    try:
        if ext in [".xls", ".xlsx"]:
            df = pd.read_excel(caminho, dtype=str)
        else:
            df = pd.read_csv(caminho, sep=None, engine="python", dtype=str)
        return df
    except Exception as e:
        print(f"  ⚠️  Erro ao ler {caminho}: {e}")
        return None


def preparar_df(df_raw):
    """Seleciona e converte colunas de interesse."""
    # Normaliza nomes de colunas
    df_raw.columns = [c.strip().lower() for c in df_raw.columns]

    colunas_presentes = [c for c in COLUNAS if c in df_raw.columns]
    df = df_raw[colunas_presentes].copy()

    # Converte vírgula decimal
    for col in df.columns:
        df[col] = df[col].astype(str).str.replace(",", ".").str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Mapeia categorias
    df["raca_cat"]   = df["q69"].map(MAPA_RACA)
    df["raca_grupo"] = df["raca_cat"].apply(
        lambda x: "Negra" if x in RACA_NEGRA else ("Não negra" if pd.notna(x) else np.nan)
    )
    df["sexo"]       = df["q7"].map(MAPA_SEXO)
    df["escolaridade"] = df["fesc"].map(MAPA_ESC)
    df["cidade_nome"]  = df["cidade"].map(MAPA_CIDADE)
    df["regiao"]       = df["cidade_nome"].map(MAPA_REGIAO)

    return df


# ── Geração dos cruzamentos ───────────────────────────────────────────────────

def gerar_dados(df_all):
    anos = sorted(df_all["ano"].dropna().unique().astype(int).tolist())
    output = {
        "anos": anos,
        "indicadores": INDICADORES,
        "cruzamentos": {}
    }

    dims = {
        "raca_grupo":   "Raça/Cor (Negra × Não negra)",
        "raca_cat":     "Raça/Cor (detalhada)",
        "sexo":         "Sexo",
        "escolaridade": "Escolaridade",
        "regiao":       "Região",
        "cidade_nome":  "Cidade",
    }

    for ind_col, ind_label in INDICADORES.items():
        if ind_col not in df_all.columns:
            print(f"  ⚠️  Indicador '{ind_col}' não encontrado, pulando.")
            continue

        output["cruzamentos"][ind_col] = {
            "label": ind_label,
            "por_ano": {},
            "dimensoes": list(dims.keys()),
        }

        for ano in anos:
            df_ano = df_all[df_all["ano"] == ano]
            output["cruzamentos"][ind_col]["por_ano"][str(ano)] = {}

            for dim_col, dim_label in dims.items():
                resultado = calcular_cruzamento(df_ano, ind_col, None, dim_col)
                output["cruzamentos"][ind_col]["por_ano"][str(ano)][dim_col] = {
                    "label": dim_label,
                    "dados": resultado,
                }

    return output


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔍 Buscando arquivos VIGITEL em:", PASTA_DADOS)

    arquivos = glob.glob(os.path.join(PASTA_DADOS, "*.xls")) + \
               glob.glob(os.path.join(PASTA_DADOS, "*.xlsx")) + \
               glob.glob(os.path.join(PASTA_DADOS, "*.csv"))

    if not arquivos:
        print(f"\n❌ Nenhum arquivo encontrado em '{PASTA_DADOS}/'")
        print("   Coloque os arquivos VIGITEL brutos lá e rode novamente.")
        return

    dfs = []
    for arq in sorted(arquivos):
        print(f"  📂 Lendo: {os.path.basename(arq)}")
        df_raw = ler_arquivo(arq)
        if df_raw is not None:
            df = preparar_df(df_raw)
            if "ano" in df.columns:
                dfs.append(df)
                anos = df["ano"].dropna().unique().astype(int)
                print(f"     ✅ {len(df):,} registros | Anos: {sorted(anos)}")

    if not dfs:
        print("\n❌ Nenhum arquivo foi carregado com sucesso.")
        return

    df_all = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Total combinado: {len(df_all):,} registros")
    print(f"   Anos disponíveis: {sorted(df_all['ano'].dropna().unique().astype(int))}")

    print("\n⚙️  Calculando cruzamentos ponderados...")
    dados = gerar_dados(df_all)

    with open(SAIDA_JSON, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Arquivo gerado: {SAIDA_JSON}")
    print(f"   Anos processados: {dados['anos']}")
    print(f"   Indicadores: {len(dados['cruzamentos'])}")
    print("\n🚀 Próximo passo: abra o index.html no navegador!")


if __name__ == "__main__":
    main()
