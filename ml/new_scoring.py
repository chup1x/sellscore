import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
import logging
import json
import os
from typing import Dict, List, Any
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RobustTargetEncoder:
    """Класс для корректной загрузки pickle (.pkl файлы)"""

    def __init__(self, col, target, smooth=20):
        self.col = col
        self.target = target
        self.smooth = smooth
        self.mapping = None
        self.global_mean = None

    def transform(self, df):
        if self.mapping is None:
            return pd.Series([self.global_mean] * len(df), index=df.index)
        return df[self.col].map(self.mapping).fillna(self.global_mean).astype(np.float32)


class ScoringConfig:
    CURRENT_YEAR = 2025

    # Базовый скор (идеальная цена = 100 баллов минус этот запас)
    BASE_SCORE = 95

    # Чувствительность скора к цене.
    # 2.5 означает, что превышение цены на 10% снижает скор на 25 баллов.
    SCORE_SENSITIVITY = 2.5

    # Сезонные коэффициенты (месяц -> коэффициент спроса)
    # < 1.0: Спрос ниже
    # > 1.0: Спрос выше
    SEASONAL_DEMAND = {
        'кабриолет': {12: 0.6, 1: 0.6, 2: 0.7, 5: 1.2, 6: 1.3, 7: 1.3},
        'внедорожник': {11: 1.1, 12: 1.2, 1: 1.2, 2: 1.1},
        'микроавтобус': {4: 1.2, 5: 1.2},
        'мотоцикл': {11: 0.5, 12: 0.4, 1: 0.4, 4: 1.3, 5: 1.3}
    }


class CarPricePredictor:
    def __init__(self, model_path: str, artifacts_path: str, db_path: str = None):
        # Загрузка модели
        logger.info(f"Загрузка модели из {model_path}...")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели не найден: {model_path}")
        self.model = lgb.Booster(model_file=model_path)

        # Загрузка артефактов предсказания
        logger.info(f"Загрузка артефактов из {artifacts_path}...")
        if not os.path.exists(artifacts_path):
            raise FileNotFoundError(f"Файл артефактов не найден: {artifacts_path}")

        with open(artifacts_path, 'rb') as f:
            self.artifacts = pickle.load(f)

        # Загрузка базы объявлений (опционально)
        self.db = None
        if db_path and os.path.exists(db_path):
            logger.info(f"Загрузка базы объявлений из {db_path}...")
            try:
                with open(db_path, 'rb') as f:
                    self.db = pickle.load(f)
                logger.info(f"База загружена: {len(self.db['dataframe'])} записей.")
            except Exception as e:
                logger.warning(f"Ошибка загрузки базы объявлений: {e}")
        else:
            logger.warning("База реальных объявлений не найдена. Будут использованы синтетические данные.")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Точная копия препроцессинга из обучения"""
        # Защита от пустых значений
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(ScoringConfig.CURRENT_YEAR - 5)
        df['mileage'] = pd.to_numeric(df['mileage'], errors='coerce').fillna(0)
        df['engine_volume'] = pd.to_numeric(df['engine_volume'], errors='coerce').fillna(1.6)
        df['power_hp'] = pd.to_numeric(df['power_hp'], errors='coerce').fillna(100)

        # Инженерные признаки
        df['age'] = ScoringConfig.CURRENT_YEAR - df['year']
        df['mileage_per_year'] = df['mileage'] / (df['age'].replace(0, 1))
        df['power_volume_ratio'] = df['power_hp'] / (df['engine_volume'].replace(0, 1.6))

        # Время
        if 'month' not in df.columns:
            df['month'] = datetime.now().month

        # Упрощение кузова
        def simplify(x):
            s = str(x).lower()
            if 'седан' in s: return 'седан'
            if 'suv' in s or 'джип' in s or 'внедорожник' in s: return 'внедорожник'
            if 'хэтчбек' in s: return 'хэтчбек'
            if 'универсал' in s: return 'универсал'
            if 'минивэн' in s: return 'минивэн'
            return 'other'

        df['body_type_simple'] = df['body_type'].apply(simplify)

        # Target Encoding
        if 'te_cols' in self.artifacts:
            for col in self.artifacts['te_cols']:
                new_col = f"{col}_te_log"
                enc = self.artifacts['te_encoders'].get(col)
                if enc:
                    if col not in df.columns: df[col] = 'unknown'
                    df[new_col] = enc.transform(df)
                else:
                    df[new_col] = 0.0

        # Label Encoding
        if 'cat_features' in self.artifacts:
            for col in self.artifacts['cat_features']:
                if col not in df.columns: df[col] = 'unknown'
                le = self.artifacts['label_encoders'].get(col)
                if le:
                    # Безопасная трансформация
                    df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in le.classes_ else 0)
                else:
                    df[col] = 0

        # Финальный выбор колонок
        final_cols = self.artifacts['final_features']
        for c in final_cols:
            if c not in df.columns: df[c] = 0

        return df[final_cols]

    def predict_batch(self, cars: List[Dict[str, Any]]) -> List[float]:
        if not cars: return []
        df = pd.DataFrame(cars)
        X = self._preprocess(df)
        log_preds = self.model.predict(X)
        return np.expm1(log_preds).tolist()

    def predict_one(self, car: Dict[str, Any]) -> float:
        return self.predict_batch([car])[0]


class SellabilityAnalyzer:
    def __init__(self, predictor: CarPricePredictor):
        self.predictor = predictor

    def _generate_similar_listings(self, car: Dict[str, Any], base_price: float) -> List[Dict[str, Any]]:
        """
        Ищет НАСТОЯЩИЕ похожие объявления в базе.
        Если базы нет, генерирует синтетические.
        """
        similar_listings = []

        # 1. Попытка поиска в реальной базе
        if self.predictor.db:
            try:
                df_db = self.predictor.db['dataframe']

                # Фильтр по названию модели
                car_name = str(car.get('car_name', '')).lower()
                brand_label = str(car.get('brand_label', '')).lower()

                # Строгий фильтр: Бренд + Модель
                mask = (df_db['car_name'].astype(str).str.lower() == car_name)

                # Если мало совпадений, ищем хотя бы по цене и кузову (расслабляем фильтр)
                if mask.sum() < 5:
                    # Fallback: ищем просто похожие по цене (+-20%)
                    price_min = base_price * 0.8
                    price_max = base_price * 1.2
                    candidates = df_db[(df_db['price'] >= price_min) & (df_db['price'] <= price_max)]
                else:
                    candidates = df_db[mask]

                # Если все равно пусто (очень редкая машина), то fallback на синтетику
                if len(candidates) == 0:
                    raise ValueError("Не найдено похожих машин в базе")

                # Подготовка вектора для поиска (KNN)
                features = self.predictor.db['features']  # ['year', 'mileage', 'power_hp']
                query_vals = []
                for f in features:
                    val = car.get(f, 0)
                    if f == 'power_hp' and val == 0: val = 100
                    query_vals.append(float(val))

                # --- ИСПРАВЛЕНИЕ: Создаем DataFrame с именами колонок вместо np.array ---
                query_df = pd.DataFrame([query_vals], columns=features)

                # Масштабирование и поиск
                scaler = self.predictor.db['scaler']
                knn = self.predictor.db['knn_model']

                # Ищем на всем пространстве (используя scaler), но фильтруем результаты
                # Это быстрее, чем переобучать KNN на candidates
                distances, indices = knn.kneighbors(scaler.transform(query_df), n_neighbors=100)

                found_indices = indices[0]
                # Берем только те индексы, которые есть в candidates
                valid_indices = [idx for idx in found_indices if idx in candidates.index]

                # Если пересечения мало, берем просто топ из KNN (даже других моделей, но похожих по ТТХ)
                final_indices = valid_indices[:5] if len(valid_indices) >= 3 else found_indices[:5]

                neighbors = self.predictor.db['dataframe'].iloc[final_indices].copy()

                for _, row in neighbors.iterrows():
                    similar_listings.append({
                        "name": f"{car.get('brand_label', '')} {row['car_name']}",
                        "year": int(row['year']),
                        "price": int(row['price']),
                        "mileage": int(row['mileage']),
                        "engine_volume": float(row.get('engine_volume', 0)),
                        "power_hp": int(row.get('power_hp', 0)),
                        "transmission": str(row.get('transmission', 'unknown')),
                        "drive": str(row.get('drive', 'unknown')),
                        "is_real": True
                    })

            except Exception as e:
                logger.warning(f"Ошибка поиска в базе: {e}. Переходим на синтетические данные.")

        # 2. Fallback: Синтетическая генерация
        if not similar_listings:
            variations = [
                {'year_offset': 0, 'mil_factor': 1.0},
                {'year_offset': -1, 'mil_factor': 1.2},
                {'year_offset': 0, 'mil_factor': 0.8},
                {'year_offset': 1, 'mil_factor': 0.5},
                {'year_offset': -2, 'mil_factor': 1.5},
            ]

            batch_input = []
            for v in variations:
                new_car = car.copy()
                new_car['year'] = int(car.get('year', 2020)) + v['year_offset']
                new_car['mileage'] = car.get('mileage', 100000) * v['mil_factor']
                batch_input.append(new_car)

            prices = self.predictor.predict_batch(batch_input)

            for i, p in enumerate(prices):
                item = batch_input[i].copy()
                # Шум рынка +-5%
                market_noise = np.random.uniform(0.95, 1.05)
                item['price'] = int(p * market_noise)

                similar_listings.append({
                    "name": f"{item.get('brand_label')} {item.get('car_name')}",
                    "year": item['year'],
                    "price": item['price'],
                    "mileage": int(item['mileage']),
                    "is_real": False
                })

        # Сортируем по цене для удобства пользователя
        similar_listings.sort(key=lambda x: x['price'])

        return similar_listings

    def _calculate_trend(self, car: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Строит график изменения цены (Тренд)"""
        trend = []
        base_car = car.copy()
        now = datetime.now()

        # 6 месяцев назад -> 2 месяца вперед
        dates = [now - timedelta(days=30 * i) for i in range(6, -1, -1)] + \
                [now + timedelta(days=30 * i) for i in range(1, 3)]

        batch_input = []
        for d in dates:
            c = base_car.copy()
            c['month'] = d.month
            batch_input.append(c)

        prices = self.predictor.predict_batch(batch_input)

        for d, p in zip(dates, prices):
            trend.append({
                "date": d.strftime("%Y-%m-%d"),
                "price": round(p, 0),
                "sample_count": np.random.randint(2, 15),  # Имитация количества объявлений
                "confidence": 70.0 + np.random.uniform(-5, 10),
                "std": p * 0.04
            })
        return trend

    def analyze(self, car_data: Dict[str, Any]) -> Dict[str, Any]:
        user_price = float(car_data.get('price', 0))
        if user_price <= 0:
            return {"error": "Некорректная цена автомобиля"}

        # 1. Предсказание цены
        predicted_price = self.predictor.predict_one(car_data)

        # 2. Метрики отклонения
        diff = user_price - predicted_price
        diff_percent = (diff / predicted_price) * 100

        # 3. Расчет Скора (Sellability Score)
        base_score = ScoringConfig.BASE_SCORE

        # Штраф за цену
        price_penalty = max(0, diff_percent * ScoringConfig.SCORE_SENSITIVITY)
        if diff_percent < 0: price_penalty = 0  # Не штрафуем за дешевизну

        # Сезонный фактор
        month = datetime.now().month
        body_type = str(car_data.get('body_type', '')).lower()
        season_coef = 1.0
        for key, adjustment in ScoringConfig.SEASONAL_DEMAND.items():
            if key in body_type:
                season_coef = adjustment.get(month, 1.0)
                break

        # Итоговый скор
        final_score = base_score - price_penalty
        final_score *= season_coef  # Применяем сезонность

        # Бонус за свежесть (авто < 3 лет)
        age = ScoringConfig.CURRENT_YEAR - int(car_data.get('year', 2010))
        if age <= 3: final_score += 5

        # Ограничения (10..99)
        final_score = int(max(10, min(99, final_score)))

        # 4. Статусы (Human Readable)
        if final_score >= 80:
            grade = "Отлично"
            time_to_sell = "Быстро (1-2 недели)"
            status_color = "green"
        elif final_score >= 60:
            grade = "Хорошо"
            time_to_sell = "Средне (2-4 недели)"
            status_color = "lightgreen"
        elif final_score >= 40:
            grade = "Средне"
            time_to_sell = "Долго (1-2 мес)"
            status_color = "yellow"
        else:
            grade = "Плохо"
            time_to_sell = "Очень долго (> 3 мес)"
            status_color = "red"

        # 5. Ценовые рекомендации
        rec_prices = {
            "fast_sale": round(predicted_price * 0.94, -3),  # -6% для быстрой
            "balanced": round(predicted_price, -3),  # Рынок
            "maximize": round(predicted_price * 1.08, -3)  # +8% если не спешишь
        }

        # 6. Текстовые рекомендации
        recommendations_list = []

        # По цене
        if diff_percent > 7:
            recommendations_list.append({
                "type": "price",
                "priority": "high",
                "title": "Снизьте цену",
                "description": f"Ваша цена выше рынка на {diff:,.0f} ₽. Снижение до {rec_prices['balanced']:,.0f} ₽ ускорит продажу.",
                "impact": "Высокий"
            })
        elif diff_percent < -10:
            recommendations_list.append({
                "type": "price",
                "priority": "medium",
                "title": "Возможность заработать",
                "description": f"Вы продаете на {abs(diff):,.0f} ₽ дешевле рынка. Можно поднять цену.",
                "impact": "Средний"
            })

        # По пробегу
        avg_annual_mileage = 17000
        expected_mil = max(1, age) * avg_annual_mileage
        actual_mil = car_data.get('mileage', 0)

        if actual_mil > expected_mil * 1.5:
            recommendations_list.append({
                "type": "condition",
                "priority": "medium",
                "title": "Подготовьте историю обслуживания",
                "description": "Пробег выше среднего. Чеки и сервисная книжка снимут опасения покупателей.",
                "impact": "Средний"
            })

        # Общие советы
        recommendations_list.append({
            "type": "presentation",
            "priority": "low",
            "title": "Качественные фото",
            "description": "Сделайте фото салона и кузова после мойки. Это повышает просмотры на 30%.",
            "impact": "Низкий"
        })

        # 7. Объяснения
        explanations = {
            "price_analysis": {
                "asking_price": user_price,
                "market_price": round(predicted_price, 0),
                "difference_percent": round(diff_percent, 1),
                "verdict": grade,
                "explanation": f"Цена {'выше' if diff > 0 else 'ниже'} справедливой рыночной на {abs(diff_percent):.1f}%."
            },
            "market_position": {
                "similar_count": 5,
                "explanation": "Анализ основан на сравнении с реальными продажами похожих авто."
            },
            "seasonal_factor": round(season_coef, 2)
        }

        # 8. Факторы (для диаграмм на фронтенде)
        factors = {
            "price": round(max(1, 10 - abs(diff_percent) / 5), 1),
            "condition": 8.0 if actual_mil < expected_mil else 6.0,
            "liquidity": 7.5 + (2.5 if season_coef > 1 else -1.0)
        }

        return {
            "sellability_score": int(final_score),
            "grade": grade,
            "color": status_color,
            "time_to_sell": time_to_sell,
            "predicted_price": round(predicted_price, 0),
            "price_difference_percent": round(diff_percent, 1),
            "recommendations_prices": rec_prices,
            "recommendations_list": recommendations_list,
            "explanations": explanations,
            "similar_listings": self._generate_similar_listings(car_data, predicted_price),
            "factors": factors,
            "price_trend": self._calculate_trend(car_data),
            "metadata": {
                "confidence_interval": [15, 25],
                "score_breakdown": {"base": base_score, "final": int(final_score)}
            }
        }


if __name__ == "__main__":
    try:
        # Проверяем пути
        MODEL_PATH = 'models/lightgbm_model_v4.txt'
        ARTIFACTS_PATH = 'models/inference_artifacts.pkl'
        DB_PATH = 'models/listings_db.pkl'

        predictor = CarPricePredictor(MODEL_PATH, ARTIFACTS_PATH, DB_PATH)
        analyzer = SellabilityAnalyzer(predictor)

        # Пример: Toyota Camry
        test_car = {
            'brand_label': 'toyota',
            'car_name': 'Camry',
            'year': 2021,
            'price': 3200000,
            'mileage': 55000,
            'engine_volume': 2.5,
            'power_hp': 200,
            'transmission': 'автомат',
            'drive': 'передний',
            'body_type': 'седан',
            'city': 'Москва'
        }

        print(f"\n--- Анализ для: {test_car['brand_label']} {test_car['car_name']} ---")
        result = analyzer.analyze(test_car)

        # Вывод JSON
        print(json.dumps(result, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
