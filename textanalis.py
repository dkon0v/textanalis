import string

# Открытие файла и чтение текста
with open('historical_text.txt', 'r') as file:
    text = file.read()

# Очистка текста от пунктуации
text = text.translate(str.maketrans('', '', string.punctuation))

# Преобразование текста в список слов
words = text.lower().split()

# Создание словаря для подсчета количества уникальных слов
word_counts = {}

# Подсчет количества каждого слова
for word in words:
    if word not in word_counts:
        word_counts[word] = 1
    else:
        word_counts[word] += 1

# Сортировка слов по количеству встречаемости
sorted_word_counts = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)

# Вывод 10 наиболее часто встречающихся слов
for i in range(10):
    print(f'{sorted_word_counts[i][0]}: {sorted_word_counts[i][1]}')