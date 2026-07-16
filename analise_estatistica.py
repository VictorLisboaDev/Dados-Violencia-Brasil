# ============================================================================
# ANÁLISE ESTATÍSTICA COMPLETA - HOMICÍDIOS BRASIL E MUNDO
# ============================================================================
# Autor: Projeto de Ciência de Dados
# Descrição: Análise exploratória, testes estatísticos e modelos preditivos

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, ttest_ind, mannwhitneyu, f_oneway, kruskal
from scipy.stats import pearsonr, spearmanr, linregress
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Configurações estéticas
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

print("="*70)
print("ANÁLISE ESTATÍSTICA COMPLETA - HOMICÍDIOS BRASIL E MUNDO")
print("="*70)

# ============================================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================================

print("\n📂 CARREGANDO DADOS...")

try:
    df_estados = pd.read_csv('dados/homicidios_brasil_estados_1999_2026.csv')
    df_regioes = pd.read_csv('dados/homicidios_brasil_regioes_1999_2026.csv')
    df_brasil = pd.read_csv('dados/homicidios_brasil_nacional_1999_2026.csv')
    df_mundo = pd.read_csv('dados/homicidios_mundo_paises_1999_2026.csv')
    df_comparacao = pd.read_csv('dados/comparacao_brasil_mundo_1999_2026.csv')
    print("✅ Dados carregados com sucesso!")
except FileNotFoundError:
    print("❌ Arquivos não encontrados! Execute primeiro o código de geração.")
    exit()

# ============================================================================
# 2. ANÁLISE DESCRITIVA COMPLETA
# ============================================================================

print("\n" + "="*70)
print("2. ANÁLISE DESCRITIVA")
print("="*70)

print("\n📊 ESTATÍSTICAS DESCRITIVAS - TAXA DE HOMICÍDIOS BRASIL (2000-2024):")
dados_historicos = df_brasil[(df_brasil['ano'] >= 2000) & (df_brasil['ano'] <= 2024)]
print(dados_historicos['taxa_homicidios'].describe())

print("\n📊 ESTATÍSTICAS DESCRITIVAS - HOMICÍDIOS POR ESTADO (2024):")
dados_2024 = df_estados[df_estados['ano'] == 2024]
print(dados_2024['taxa_homicidios'].describe())

print("\n📊 ESTATÍSTICAS DESCRITIVAS - COMPARAÇÃO MUNDIAL (2024):")
dados_mundo_2024 = df_mundo[df_mundo['ano'] == 2024]
print(dados_mundo_2024['taxa_homicidios'].describe())

# Estatísticas por região
print("\n📊 MÉDIA DA TAXA POR REGIÃO (2024):")
media_regiao = df_estados[df_estados['ano'] == 2024].groupby('regiao')['taxa_homicidios'].agg(['mean', 'std', 'min', 'max']).round(1)
print(media_regiao)

# ============================================================================
# 3. ANÁLISE DE TENDÊNCIA TEMPORAL
# ============================================================================

print("\n" + "="*70)
print("3. ANÁLISE DE TENDÊNCIA TEMPORAL")
print("="*70)

# Teste de tendência usando correlação de Spearman (não paramétrica)
anos_analise = df_brasil['ano'].values
taxas_brasil = df_brasil['taxa_homicidios'].values

correlacao, p_valor = spearmanr(anos_analise, taxas_brasil)
print(f"\n📈 CORRELAÇÃO DE SPEARMAN (Ano vs Taxa Brasil):")
print(f"  Coeficiente: {correlacao:.3f}")
print(f"  P-valor: {p_valor:.4f}")
if p_valor < 0.05:
    print(f"  ✅ Tendência significativa (p < 0.05)")
    if correlacao > 0:
        print(f"  📈 Tendência de AUMENTO ao longo do tempo")
    else:
        print(f"  📉 Tendência de DIMINUIÇÃO ao longo do tempo")
else:
    print(f"  ❌ Sem tendência significativa")

# Análise de quebra estrutural (mudança de padrão)
print("\n🔍 ANÁLISE DE PONTOS DE MUDANÇA (Breakpoints):")
anos_pontos = [2002, 2008, 2014, 2018, 2022]
for ano_ponto in anos_pontos:
    antes = df_brasil[df_brasil['ano'] < ano_ponto]['taxa_homicidios'].mean()
    depois = df_brasil[df_brasil['ano'] >= ano_ponto]['taxa_homicidios'].mean()
    variacao = ((depois - antes) / antes) * 100
    print(f"  {ano_ponto}: Antes={antes:.1f} → Depois={depois:.1f} (Δ {variacao:+.1f}%)")

# Decomposição de série temporal
print("\n📉 DECOMPOSIÇÃO DA SÉRIE TEMPORAL:")
serie_brasil = df_brasil.set_index('ano')['taxa_homicidios']
try:
    decomposicao = seasonal_decompose(serie_brasil, model='additive', period=5)
    print("  ✅ Decomposição realizada com sucesso!")
    print(f"  Tendência geral: {'Crescente' if decomposicao.trend.iloc[-1] > decomposicao.trend.iloc[0] else 'Decrescente'}")
except:
    print("  ⚠️ Série muito curta para decomposição sazonal")

# ============================================================================
# 4. TESTES DE NORMALIDADE
# ============================================================================

print("\n" + "="*70)
print("4. TESTES DE NORMALIDADE")
print("="*70)

# Shapiro-Wilk para distribuição das taxas em 2024
dados_test = df_estados[df_estados['ano'] == 2024]['taxa_homicidios']
stat, p_value = shapiro(dados_test)

print(f"\n📊 TESTE DE SHAPIRO-WILK (Taxas 2024):")
print(f"  Estatística: {stat:.4f}")
print(f"  P-valor: {p_value:.4f}")
if p_value > 0.05:
    print("  ✅ Distribuição NORMAL (não rejeita H0)")
else:
    print("  ❌ Distribuição NÃO-NORMAL (rejeita H0)")

# Teste de normalidade para comparação mundial
dados_mundo_2024_test = df_mundo[df_mundo['ano'] == 2024]['taxa_homicidios']
stat_m, p_value_m = shapiro(dados_mundo_2024_test)
print(f"\n📊 TESTE DE SHAPIRO-WILK (Taxas Mundo 2024):")
print(f"  Estatística: {stat_m:.4f}")
print(f"  P-valor: {p_value_m:.4f}")
if p_value_m > 0.05:
    print("  ✅ Distribuição NORMAL")
else:
    print("  ❌ Distribuição NÃO-NORMAL")

# ============================================================================
# 5. ANÁLISE DE VARIÂNCIA (ANOVA)
# ============================================================================

print("\n" + "="*70)
print("5. ANÁLISE DE VARIÂNCIA (ANOVA)")
print("="*70)

# ANOVA para comparar regiões em 2024
regioes = df_estados[df_estados['ano'] == 2024]['regiao'].unique()
grupos = [df_estados[(df_estados['ano'] == 2024) & (df_estados['regiao'] == r)]['taxa_homicidios'] for r in regioes]

# Teste paramétrico (ANOVA)
f_stat, p_anova = f_oneway(*grupos)
print(f"\n📊 ANOVA (Comparação entre Regiões 2024):")
print(f"  Estatística F: {f_stat:.4f}")
print(f"  P-valor: {p_anova:.4f}")
if p_anova < 0.05:
    print("  ✅ Diferença SIGNIFICATIVA entre regiões")
    print("  📍 As regiões têm perfis de violência distintos")
else:
    print("  ❌ Sem diferença significativa entre regiões")

# Teste não-paramétrico (Kruskal-Wallis) como complemento
h_stat, p_kruskal = kruskal(*grupos)
print(f"\n📊 KRUSKAL-WALLIS (Teste não-paramétrico):")
print(f"  Estatística H: {h_stat:.4f}")
print(f"  P-valor: {p_kruskal:.4f}")
if p_kruskal < 0.05:
    print("  ✅ Confirma diferença significativa entre regiões")
