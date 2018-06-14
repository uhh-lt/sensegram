from disambiguator import WSD

wsd = WSD("../model/wsi/cc.ru.300.vec.gz.top50.wsi-inventory.tsv", language="ru", verbose=True)

target_word = "замок"
context = "Замок Нойшванштайн буквально романтический замок баварского короля Людвига II около городка Фюссен и замка Хоэншвангау в юго-западной Баварии, недалеко от австрийской границы. Одно из самых популярных среди туристов мест на юге Германии."
max_context_words = 5
print(wsd.disambiguate(context, target_word, max_context_words))
print(wsd.get_best_sense_id(context, target_word, max_context_words))