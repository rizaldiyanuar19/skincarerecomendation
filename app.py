import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import time

# ===================================================================
# 1. KONFIGURASI HALAMAN (WAJIB PERINTAH PERTAMA)
# ===================================================================
st.set_page_config(page_title="Pencari Produk Kecantikan", layout="wide")


# ===================================================================
# 2. CSS KUSTOM UNTUK KARTU HASIL (Tampilan Rapih)
# ===================================================================
st.markdown("""
    <style>
        .result-card {
            border: 1px solid #e0e0e0;
            border-radius: 10px;
            padding: 16px;
            margin-bottom: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            transition: box-shadow 0.3s ease-in-out;
        }
        .result-card:hover {
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }
        .result-card h3 {
            margin-top: 0;
            margin-bottom: 8px;
            color: #333;
            font-size: 1.25em; /* Sedikit lebih besar */
        }
        .result-card .brand {
            font-weight: bold;
            color: #555;
            display: block; /* Membuat brand di baris baru */
            margin-bottom: 8px;
        }
        .result-card .price {
            font-size: 1.15em;
            color: #D6336C; /* Warna pink/magenta yang cocok */
            font-weight: bold;
            margin-right: 10px;
        }
        .result-card .score {
            font-size: 0.9em;
            color: #777;
        }
        .result-card a {
            text-decoration: none;
            color: #007bff;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


# ===================================================================
# 3. MEMUAT DATASET DAN MEMBANGUN MODEL (DENGAN CACHING)
# ===================================================================

@st.cache_resource
def load_all_models_and_data():
    """
    Memuat semua dataset dan membangun semua model.
    Fungsi ini hanya akan berjalan sekali.
    """
    print("Membaca dataset...")
    try:
        # Pastikan file CSV ada di folder yang sama dengan app.py
        df = pd.read_csv("sephora_data_bersih.csv", on_bad_lines='skip', engine='python')
        ratings = pd.read_csv("simulasi_rating.csv")
        print("✅ Dataset berhasil dimuat.")
    except FileNotFoundError as e:
        st.error(f"❌ ERROR: File data tidak ditemukan. Pastikan 'sephora_data_bersih.csv' dan 'simulasi_rating.csv' ada di folder yang sama. Detail: {e}")
        return None, None, None, None, None 

    print("\nMembangun model Content-Based...")
    if not df.empty:
        df['corpus_cleaned'] = df['corpus_cleaned'].fillna('')
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(df['corpus_cleaned'])
        cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        print("✅ Model Content-Based (TF-IDF) siap.")
    else:
        print("⚠️ Peringatan: DataFrame 'df' kosong.")
        return None, None, None, None, None

    print("\nMembangun model Collaborative Filtering...")
    if not df.empty and not ratings.empty:
        valid_product_ids = df['id'].unique()
        ratings_filtered = ratings[ratings['product_id'].isin(valid_product_ids)]
        if not ratings_filtered.empty:
            pivot_table = ratings_filtered.pivot_table(index='product_id', columns='user_id', values='rating').fillna(0)
            knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
            knn_model.fit(pivot_table)
            print("✅ Model Collaborative Filtering (KNN) siap.")
        else:
            pivot_table = pd.DataFrame()
            print("⚠️ Peringatan: Data rating tidak ada. Model Collaborative tidak akan aktif.")
    else:
        pivot_table = pd.DataFrame()
        print("⚠️ Peringatan: 'df' atau 'ratings' kosong. Model Collaborative tidak dibangun.")

    print("🎉 SEMUA MODEL SIAP.")
    return df, pivot_table, knn_model, cosine_sim_matrix, tfidf_matrix

# Memuat model saat aplikasi dimulai
with st.spinner('Memuat data dan model... Ini mungkin butuh beberapa saat...'):
    df, pivot_table, knn_model, cosine_sim_matrix, tfidf_matrix = load_all_models_and_data()

if df is None:
    st.error("Pemuatan data gagal. Aplikasi tidak dapat dilanjutkan.")
    st.stop()


# ===================================================================
# 4. FUNGSI REKOMENDASI 1: CARI PRODUK (Peringkat Hibrida)
# ===================================================================
def get_recommendations(keyword, top_n=10):
    """
    Fungsi rekomendasi (Peringkat Pencarian Hibrida)
    [DENGAN OPTIMISASI PERFORMA]
    """
    print(f"\n🔎 Mencari peringkat untuk keyword: '{keyword}'...")

    mask = df['name'].str.contains(keyword, case=False, na=False) | \
           df['brand'].str.contains(keyword, case=False, na=False)
    matched_products = df[mask]

    if matched_products.empty:
        return f"❌ Produk dengan keyword '{keyword}' tidak ditemukan."

    # OPTIMISASI PERFORMA
    if 'number_of_reviews' in df.columns and len(matched_products) > 50:
        print(f"⚠️ Hasil terlalu banyak ({len(matched_products)}), mengambil top 50 populer...")
        matched_products = matched_products.sort_values('number_of_reviews', ascending=False).head(50)
    elif 'rating' in df.columns and len(matched_products) > 50:
         print(f"⚠️ Hasil terlalu banyak ({len(matched_products)}), mengambil top 50 rating tertinggi...")
         matched_products = matched_products.sort_values('rating', ascending=False).head(50)

    results = []
    for _, product_row in matched_products.iterrows():
        product_id = product_row['id']
        product_idx = product_row.name
        sim_scores = sorted(list(enumerate(cosine_sim_matrix[product_idx])), key=lambda x: x[1], reverse=True)
        top_sim_scores = sim_scores[1:6]
        content_score = np.mean([score for _, score in top_sim_scores]) if top_sim_scores else 0
        collab_score = 0
        is_new_product = pivot_table.empty or product_id not in pivot_table.index
        if not is_new_product:
            try:
                distances, _ = knn_model.kneighbors(pivot_table.loc[product_id].values.reshape(1, -1), n_neighbors=6)
                collab_score = 1 - np.mean(distances[0][1:]) if len(distances[0]) > 1 else 0
            except Exception:
                is_new_product = True
        if is_new_product:
            w_content, w_collab = 1.0, 0.0
        else:
            w_content, w_collab = 0.6, 0.4
        hybrid_score = (w_content * content_score) + (w_collab * collab_score)
        results.append({
            'name': product_row['name'],
            'brand': product_row['brand'],
            'price': product_row['price'],
            'url': product_row['URL'],
            'score': round(hybrid_score, 4)
        })
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
    return pd.DataFrame(sorted_results[:top_n])

# ===================================================================
# 5. FUNGSI REKOMENDASI 2: PRODUK SERUPA
# ===================================================================
def get_similar_products(keyword, top_n=10):
    """
    Fungsi rekomendasi (Produk Serupa Hibrida)
    """
    print(f"\n🔎 Mencari produk serupa untuk keyword: '{keyword}'...")

    mask = df['name'].str.contains(keyword, case=False, na=False) | \
           df['brand'].str.contains(keyword, case=False, na=False)
    matched_products = df[mask]

    if matched_products.empty:
        return f"❌ Produk dengan keyword '{keyword}' tidak ditemukan."

    seed_product = matched_products.iloc[0]
    seed_idx = seed_product.name
    seed_id = seed_product['id']
    original_matched_ids = set(matched_products['id'])

    print(f"🌱 Menggunakan seed product: '{seed_product['name']}' (Brand: {seed_product['brand']})")

    is_new_product = pivot_table.empty or seed_id not in pivot_table.index
    if is_new_product:
        print("❄️ (Produk seed adalah item cold start, hanya menggunakan Content-Based)")
        w_content, w_collab = 1.0, 0.0
    else:
        w_content, w_collab = 0.6, 0.4

    k_candidates = 50
    sim_scores_content = list(enumerate(cosine_sim_matrix[seed_idx]))
    sorted_scores_content = sorted(sim_scores_content, key=lambda x: x[1], reverse=True)[1:k_candidates+1]

    content_candidates = {}
    for i, score in sorted_scores_content:
        product_id = df.iloc[i]['id']
        content_candidates[product_id] = score

    collab_candidates = {}
    if not is_new_product:
        try:
            distances, indices = knn_model.kneighbors(
                pivot_table.loc[seed_id].values.reshape(1, -1),
                n_neighbors=k_candidates + 1
            )
            neighbor_scores = [1 - d for d in distances[0][1:]]
            neighbor_indices = indices[0][1:]
            for i, score in zip(neighbor_indices, neighbor_scores):
                product_id = pivot_table.index[i]
                collab_candidates[product_id] = score
        except Exception as e:
            print(f"⚠️ Peringatan saat mengambil data collab: {e}")

    all_candidates = set(content_candidates.keys()) | set(collab_candidates.keys())
    hybrid_scores = []
    for prod_id in all_candidates:
        if prod_id in original_matched_ids:
            continue
        content_score = content_candidates.get(prod_id, 0)
        collab_score = collab_candidates.get(prod_id, 0)
        hybrid_score = (w_content * content_score) + (w_collab * collab_score)
        hybrid_scores.append({'id': prod_id, 'score': round(hybrid_score, 4)})

    if not hybrid_scores:
         return f"✅ Produk seed ditemukan, namun tidak ada produk serupa yang relevan."

    sorted_results = sorted(hybrid_scores, key=lambda x: x['score'], reverse=True)[:top_n]
    result_ids = [rec['id'] for rec in sorted_results]
    final_df = df[df['id'].isin(result_ids)].copy()
    scores_map = {rec['id']: rec['score'] for rec in sorted_results}
    final_df['score'] = final_df['id'].map(scores_map)
    final_df = final_df.sort_values('score', ascending=False)
    return final_df[['name', 'brand', 'price', 'URL', 'score']]

# ===================================================================
# 6. FUNGSI HELPER: Untuk Menampilkan Hasil (Tampilan Rapih)
# ===================================================================
def display_results(results, success_message):
    """
    Menampilkan hasil (DataFrame) dalam format kartu kustom.
    """
    if isinstance(results, pd.DataFrame):
        st.success(success_message)
        
        if results.empty:
            st.info("Tidak ada hasil yang ditemukan untuk ditampilkan.")
            return
            
        # Iterasi melalui baris dataframe dan tampilkan sebagai kartu
        for index, row in results.iterrows():
            # Mengganti 'URL' dengan 'url' jika 'URL' tidak ada, atau sebaliknya
            # Ini untuk memastikan fleksibilitas jika nama kolom berubah
            url = row.get('URL', row.get('url', '#'))
            
            st.markdown(f"""
                <div class="result-card">
                    <h3>{row.get('name', 'Nama Tidak Tersedia')}</h3>
                    <span class="brand">{row.get('brand', 'Brand Tidak Tersedia')}</span>
                    <div>
                        <span class="price">Rp {row.get('price', 'N/A')}</span>
                        <span class="score">(Skor: {row.get('score', 0):.4f})</span>
                    </div>
                    <a href="{url}" target="_blank">Lihat Produk</a>
                </div>
            """, unsafe_allow_html=True)
    
    else:
        # Menampilkan pesan error, misal: "Produk... tidak ditemukan."
        st.warning(results) 

# ===================================================================
# 7. ANTARMUKA PENGGUNA (UI) STREAMLIT (Tampilan Baru)
# ===================================================================

st.title("Jelajahi Produk Kecantikan 💄")
st.write(
    "Temukan produk skincare dan makeup terbaik untuk Anda. "
    "Ketik nama produk atau brand di bawah ini."
)

# Input dari pengguna
keyword = st.text_input(
    "Masukkan nama produk atau brand (Contoh: 'Dior', 'Gucci', 'Serum'):",
    "Dior" # Nilai default
)

# Membuat dua tombol bersebelahan
col1, col2 = st.columns([1, 1])

with col1:
    # Tombol 1: Peringkat Pencarian (Diganti namanya)
    if st.button("Cari Produk", type="primary"): # type="primary" membuatnya menonjol
        if not keyword:
            st.warning("Silakan masukkan keyword.")
        else:
            with st.spinner(f"Mencari produk untuk '{keyword}'..."):
                time.sleep(0.5) 
                results = get_recommendations(keyword, top_n=10)
                display_results(results, f"Menampilkan Peringkat Teratas untuk '{keyword}':")

with col2:
    # Tombol 2: Produk Serupa (Diganti namanya)
    if st.button("Temukan yang Serupa"):
        if not keyword:
            st.warning("Silakan masukkan keyword.")
        else:
            with st.spinner(f"Mencari produk yang mirip dengan '{keyword}'..."):
                time.sleep(0.5)
                results = get_similar_products(keyword, top_n=10)
                display_results(results, f"Menampilkan Produk Serupa dengan '{keyword}':")

st.caption("Aplikasi ini didukung oleh model rekomendasi hibrida (Content-Based & Collaborative).")