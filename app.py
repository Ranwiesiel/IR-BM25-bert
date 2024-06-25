from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])


def query():
    with open('bert_embeddings.pkl', 'rb') as f:
        df = pickle.load(f)
    with open('bm25_model.pkl', 'rb') as f:
        bm25 = pickle.load(f)
    with open('data/json/all_meta.json') as f:
        meta_data = [json.loads(line) for line in f]

    bm25_model = bm25['bm25']
    bm25Id = bm25['id']

    default_verdicts = ["guilty", "lepas", "bebas"]
    results = []
    
    try:
        meta_df = pd.DataFrame(meta_data)
        user_query = request.form.get('query').lower()
        get_filter = request.form.get('verdict')
        # print(get_filter)

        if get_filter == '':
            meta_df_filtered = meta_df[meta_df['verdict'].isin(default_verdicts)]
        else:
            meta_df_filtered = meta_df[meta_df['verdict'] == get_filter]

        tokenized_query = user_query.split()
        scores = bm25_model.get_scores(tokenized_query)
        scores = list(zip(bm25Id, scores))
        filtered_scores = [(id, score) for id, score in scores if id in meta_df_filtered['id'].values]

        filtered_scores.sort(key=lambda x: x[1], reverse=True)

        if not filtered_scores:
            print("No matching documents found after filtering.")
        else:
            # best_id = filtered_scores[0][0]
            for idx, (best_id, score) in enumerate(filtered_scores):
                df_filtered = df[df['id'] == best_id]
                meta_df_best = meta_df_filtered[meta_df_filtered['id'] == best_id]

                best_doc = pd.merge(df_filtered, meta_df_best, on='id')
        # best_idx = scores.argmax()
        # df = pd.merge(df, meta_df, on='id')
        # best_doc = df.iloc[best_idx]

                result = {
                    'id': best_doc['id'].values[0],
                    'text': best_doc['text'].values[0],
                    'verdict': best_doc['verdict'].values[0],
                    'indictment': best_doc['indictment'].values[0],
                    'lawyer': best_doc['lawyer'].values[0],
                    'owner': best_doc['owner'].values[0],
                    'score': score
                }

                if score <= 0:
                    return render_template('results.html', results=results)

                results.append(result)
                print(f"Result {idx + 1}")
    
    except Exception as e:
        print(f"Error: {e}")
        return render_template('index.html', error="Error occurred while fetching results.")
    

if __name__ == '__main__':
    app.run(debug=True)
