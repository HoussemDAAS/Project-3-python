import pandas as pd
from sqlalchemy import create_engine
from collections import Counter
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    sns.set(style="whitegrid")
except Exception:
    sns = None

try:
    from wordcloud import WordCloud, STOPWORDS
except Exception:
    WordCloud = None
    STOPWORDS = set()


# ================================
# Configuration base de données
# ================================
DB_USERNAME = 'root'
DB_PASSWORD = ''
DB_HOST = 'localhost'
DB_PORT = '3306'
DB_NAME = 'dataControl'

CONNECTION_STRING = f"mysql+pymysql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONNECTION_STRING)


# ================================
# Utilitaires
# ================================
def safe_text_column(df):
    """Retourne la meilleure colonne texte disponible pour l'analyse."""
    for col in ["text_processed", "text_cleaned", "text"]:
        if col in df.columns:
            return col
    raise ValueError("Aucune colonne de texte trouvée (text_processed, text_cleaned, text)")


def load_table(table_name: str) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {table_name}", engine)


def map_news_label_to_class(label: str) -> str:
    """Mappe les labels du dataset news vers classes normalisées."""
    if label is None:
        return "normal"
    label = str(label).strip().lower()
    return "fake" if label == "fake" else "normal"


def map_labeled_class_to_class(cls) -> str:
    """Mappe les classes 0/1/2 du dataset labeled vers {hate, normal}.
    - 0: hate_speech -> hate
    - 1: offensive_language -> hate (regroupé)
    - 2: neither -> normal
    """
    try:
        cls_int = int(cls)
    except Exception:
        return "normal"
    return "hate" if cls_int in (0, 1) else "normal"


def plot_class_distribution(df: pd.DataFrame, class_col: str, title: str):
    counts = df[class_col].value_counts().sort_index()
    print(f"\n📊 Distribution des classes pour {title}:")
    print(counts)
    plt.figure(figsize=(6, 4))
    if sns is not None:
        sns.barplot(x=counts.index, y=counts.values, palette="deep")
    else:
        plt.bar(counts.index, counts.values, color="#4C78A8")
    plt.title(title)
    plt.xlabel("Classe")
    plt.ylabel("Nombre")
    plt.tight_layout()
    plt.show()


def generate_wordcloud(texts, title: str, max_words: int = 200):
    if WordCloud is None:
        print("⚠️ Le package 'wordcloud' n'est pas installé. Exécutez: pip install wordcloud")
        return
    combined_text = " ".join(texts)
    wc = WordCloud(width=900, height=500, background_color="white", stopwords=STOPWORDS, max_words=max_words).generate(combined_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def lexical_stats(df: pd.DataFrame, text_col: str, title: str, top_n: int = 20):
    lengths = df[text_col].astype(str).str.split().apply(len)
    avg_len = lengths.mean()
    print(f"\n🧮 Statistiques lexicales ({title}):")
    print(f"- Longueur moyenne (mots): {avg_len:.2f}")

    # Fréquences de mots (excluant stopwords simples)
    stop = set(STOPWORDS) | {"rt", "amp", "https", "http", "co"}
    words = (word for text in df[text_col].astype(str) for word in text.split())
    filtered = [w for w in words if len(w) > 2 and w.lower() not in stop]
    freq = Counter(filtered).most_common(top_n)
    print("- Top mots-clés:")
    for w, c in freq:
        print(f"  {w}: {c}")

    # Plot
    if freq:
        labels, values = zip(*freq)
        plt.figure(figsize=(10, 4))
        if sns is not None:
            sns.barplot(x=list(labels), y=list(values), palette="muted")
        else:
            plt.bar(labels, values, color="#F58518")
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Top {top_n} mots-clés - {title}")
        plt.xlabel("Mot")
        plt.ylabel("Fréquence")
        plt.tight_layout()
        plt.show()


SENSITIVE_KEYWORDS = {
    # Français
    "haine", "insulte", "injure", "mensonge", "menteur", "raciste", "violence",
    # Anglais
    "hate", "insult", "lie", "liar", "racist", "violence", "kill", "bitch", "fuck"
}


def detect_sensitive_words(df: pd.DataFrame, text_col: str, title: str, keywords=SENSITIVE_KEYWORDS):
    print(f"\n🚨 Détection de mots sensibles ({title})")
    counts = {k: 0 for k in keywords}
    for text in df[text_col].astype(str):
        tokens = text.split()
        for k in keywords:
            counts[k] += sum(1 for t in tokens if t.lower() == k)
    # Trier et afficher
    sorted_counts = sorted(((k, v) for k, v in counts.items() if v > 0), key=lambda x: x[1], reverse=True)
    if not sorted_counts:
        print("- Aucun mot sensible détecté dans l'échantillon.")
        return
    for k, v in sorted_counts:
        print(f"  {k}: {v}")
    # Plot
    labels, values = zip(*sorted_counts)
    plt.figure(figsize=(8, 4))
    if sns is not None:
        sns.barplot(x=list(labels), y=list(values), palette="dark")
    else:
        plt.bar(labels, values, color="#54A24B")
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Mots sensibles - {title}")
    plt.xlabel("Mot")
    plt.ylabel("Occurrences")
    plt.tight_layout()
    plt.show()


def run_analysis():
    print("=" * 70)
    print("ETAPE 2 : ANALYSE DE DONNÉES")
    print("=" * 70)

    # Charger données
    df_news = load_table("news")
    df_labeled = load_table("labeled")

    text_news = safe_text_column(df_news)
    text_labeled = safe_text_column(df_labeled)

    # Classes normalisées
    if "label" in df_news.columns:
        df_news["class_norm"] = df_news["label"].apply(map_news_label_to_class)
    else:
        df_news["class_norm"] = "normal"

    if "class" in df_labeled.columns:
        df_labeled["class_norm"] = df_labeled["class"].apply(map_labeled_class_to_class)
    else:
        df_labeled["class_norm"] = "normal"

    # Distribution des classes
    plot_class_distribution(df_news, "class_norm", "Distribution - NEWS (fake/normal)")
    plot_class_distribution(df_labeled, "class_norm", "Distribution - LABELED (hate/normal)")

    # Word clouds (échantillon pour performance)
    sample_news = df_news[text_news].dropna().astype(str).head(10000)
    sample_labeled = df_labeled[text_labeled].dropna().astype(str).head(10000)
    generate_wordcloud(sample_news, "Word Cloud - NEWS")
    generate_wordcloud(sample_labeled, "Word Cloud - LABELED")

    # Statistiques lexicales
    lexical_stats(df_news, text_news, "NEWS")
    lexical_stats(df_labeled, text_labeled, "LABELED")

    # Détection de mots sensibles
    detect_sensitive_words(df_news, text_news, "NEWS")
    detect_sensitive_words(df_labeled, text_labeled, "LABELED")

    # Vue combinée
    combined = pd.DataFrame({
        "text": pd.concat([df_news[text_news], df_labeled[text_labeled]], ignore_index=True),
        "class_norm": pd.concat([df_news["class_norm"], df_labeled["class_norm"]], ignore_index=True),
    })
    plot_class_distribution(combined, "class_norm", "Distribution - COMBINÉ (fake/hate/normal)")
    lexical_stats(combined, "text", "COMBINÉ")
    generate_wordcloud(combined["text"].dropna().astype(str).head(15000), "Word Cloud - COMBINÉ")
    detect_sensitive_words(combined, "text", "COMBINÉ")

    print("\n✅ Étape 2 terminée : analyses générées.")


if __name__ == "__main__":
    run_analysis()