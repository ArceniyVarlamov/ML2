# Данные для пайплайна

Ожидаемая структура (конфиги в `configs/` ссылаются на эти пути):

```
Data/
  Train/
    train_main_features.parquet
    train_extra_features.parquet
  Test/
    test_main_features.parquet
    test_extra_features.parquet
  Main/
    train_target.parquet
    sample_submit.parquet
```

Положите сюда файлы датасета (скачанные с площадки соревнования или из бэкапа).  
Папка **Data/** в репозитории не игнорируется — можно коммитить файлы при необходимости.
