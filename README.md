# hse_nis
Project repository for HSE NIS. Homework task for fraud detection.

*Результаты скоринга в письме. А так же в папке predict_for_github*

*Описание основных ноутбуков:*

Обработка данных:

- Первоначально база была разбита на части и переведена из формата sqlite в формат feather (аналог csv).
Это было сделано в ноутбуках data_preparation/alex_raw_split*
- Отдельно были проанализированы whois записи ip адресов на предмет вхождения mts/megafon/tele2/vimpelcom в ноутбуке /data_preparation/ip_whois.ipynb
- Затем была произведена генерация признаков и *_дополнительных признаков_* в ноутбуках data_preparation/alex_data_prep_for_pipeline_temp-v3*
- woe на признаки в ноутбуке /data_preparation/add_woe_ildar.ipynb
- *_квантили_* были подсчитаны в ноутбуке /data_preparation/quantiles_fast_template_ildar.ipynb
- в дальнейшем все признаки обьединялись в 1 датафреймы для train / test в ноутбуках /data_preparation/combine_datasets_v1*

Построение моделей:

- *Финальные предсказания* делались в prediction/final_prediction.ipynb. Взяли обычный xgboost, обучили его на последних двух месяцах и предсказали.
- *Задание про indexer* и задание про *применить один из методов работы с несбалансированными классами* в ноутбуке indexer_undersampling.ipynb. В ноутбуке реализована forget model, использован SMOTE и сделан undersampling данных.
- Подбор гиперпараметров xgboost в ноутбуке model_and_validation-V1_withoutCV_Simakov_Shvedov


Визуализация:

- *Precision-Recall curve* notebooks/pr_curve.ipynb
- *Задание про SHAP* exploration/shap-visualization.ipynb
- Большая визуализация данных в exploration/andrey_data_visualization.ipynb
