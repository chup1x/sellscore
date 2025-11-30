from flask import Flask, request, jsonify, render_template_string
from new_scoring import SellabilityAnalyzer, CarPricePredictor
import os

app = Flask(__name__)

MODEL_PATH = 'models/lightgbm_model_v4.txt'
ARTIFACTS_PATH = 'models/inference_artifacts.pkl'


def _init_analyzer() -> SellabilityAnalyzer | None:
    try:
        print(f"[INIT] Loading model from {MODEL_PATH} and artifacts from {ARTIFACTS_PATH}")
        if not os.path.exists(MODEL_PATH):
            print(f"[INIT][WARNING] Model file not found at {MODEL_PATH}")
        if not os.path.exists(ARTIFACTS_PATH):
            print(f"[INIT][WARNING] Artifacts file not found at {ARTIFACTS_PATH}")

        predictor = CarPricePredictor(MODEL_PATH, ARTIFACTS_PATH)
        analyzer_instance = SellabilityAnalyzer(predictor)

        print('[INIT] SellabilityAnalyzer initialized successfully')
        return analyzer_instance
    except Exception as exc:
        print(f"[INIT][ERROR] Failed to initialize analyzer: {exc}")
        import traceback
        traceback.print_exc()
        return None


ANALYZER = _init_analyzer()


def _post_process_result(result, user_data):
    """
    Исправляет данные перед отправкой на клиент:
    1. Досчитывает cheaper_count/more_expensive_count
    2. Заполняет пропуски в similar_listings
    3. Гарантирует наличие difference в price_analysis
    """
    # 1. Fix Market Position (считаем сами, так как скоринг это не возвращает)
    if 'explanations' in result and 'market_position' in result['explanations']:
        mp = result['explanations']['market_position']
        user_price = user_data.get('price', 0)
        similar = result.get('similar_listings', [])

        if user_price and similar:
            cheaper = sum(1 for s in similar if s.get('price', 0) < user_price)
            expensive = sum(1 for s in similar if s.get('price', 0) > user_price)
            mp['cheaper_count'] = cheaper
            mp['more_expensive_count'] = expensive
        else:
            mp['cheaper_count'] = mp.get('cheaper_count', 0)
            mp['more_expensive_count'] = mp.get('more_expensive_count', 0)

    # 2. Fix Similar Listings (убираем undefined)
    required_keys = ['transmission', 'drive', 'body_type', 'color', 'engine_type']
    for item in result.get('similar_listings', []):
        # Заполняем строковые поля
        for k in required_keys:
            if k not in item or item[k] is None:
                # Пытаемся взять из данных пользователя или ставим дефолт
                val = user_data.get(k)
                item[k] = val if val else 'Не указано'

        # Заполняем числовые поля, если они N/A
        if 'engine_volume' not in item or not item['engine_volume']:
            item['engine_volume'] = user_data.get('engine_volume', 0.0)
        if 'power_hp' not in item or not item['power_hp']:
            item['power_hp'] = user_data.get('power_hp', 0)

    # 3. Fix Price Analysis Difference
    if 'explanations' in result and 'price_analysis' in result['explanations']:
        pa = result['explanations']['price_analysis']
        if 'difference' not in pa or pa['difference'] is None:
            # Пересчитываем разницу, если она потерялась
            ask = pa.get('asking_price', user_data.get('price', 0))
            mkt = pa.get('market_price', 0)
            if mkt > 0:
                diff_p = ((ask - mkt) / mkt) * 100
                pa['difference'] = round(diff_p, 1)
            else:
                pa['difference'] = 0.0

    return result

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        if not data: return jsonify({'error': 'No data'}), 400

        # Map asking_price -> price
        if 'asking_price' in data and 'price' not in data:
            data['price'] = data['asking_price']

        # Set Defaults for fields new_scoring might need but form doesn't send
        defaults = {
            'body_type': 'седан',
            'drive': 'передний',
            'color': 'не указан',
            'city': 'Москва',
            'owner_count': 1
        }
        for k, v in defaults.items():
            if k not in data or not data[k]:
                data[k] = v

        if ANALYZER is None:
            return jsonify({'error': 'Analyzer not ready'}), 503

        print(f"[INFO] Analyzing: {data.get('brand_label')} {data.get('car_name')}")
        result = ANALYZER.analyze(data)

        # --- VITAL FIX: Post-process the result to fix "undefined" ---
        result = _post_process_result(result, data)

        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
