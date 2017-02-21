"Хотите удивительную историю? Маленького Сашу отец часто посылал в магазин за мандаринами - давал ему пару сотен рублей, а маленький Саша возвращался с мандаринами и сдачей. Как-то раз отец что-то перепутал, и вместо ста рублей выдал сто индонезийских рупий. Ну всякое бывает, спросите вы, что ж тут удивительного. А вот что: маленький Саша сразу же понял, что тут что-то не так!!!"

13.02.17 - 18.02.17
Интересная постановка задачи - для обучения дана выборка из некоторого распределения, затем подаются новые точки из него же и какие-то левые, нужно отделять первые от вторых.

[Статья по Isolation Forest](http://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
Краткий перессказ статьи - Isolation Forest круче всех. Работая в предположении "аномальные точки - изолированные, то есть с низкой плотностью априорного распределения", можно за линейное время (с константой, не пропорциональной размерности пространства) научиться выдавать объектам адекватный скор аномальности, превосходящей по метрике AUC алгоритмы-предшественники (причём, якобы, особенно в многомерном случае - почему так, мне непонятно).

Ознакомился с Isolation Forest (Isolation Forest First Look.ipynb) и его sklearn-овской реализацией. Появилось страстное желание сделать для каждого дерева случайный выбор базиса (позже: это называется умными словами rotated bagging). Проведены эксперименты для разного числа деревьев в следующих случаях: два кластера; один кластер; два кластера с аномалиями в обучении; кластер в форме синуса. Во всех случаях контуры одинакового скора аномальности становились эстетически красивее. По мере auc-roc эти искусственные задачи Isolation Forest решает с высоким результатом и за линейное время, однако выбор адекватной метрики оценивания качества работы детектора аномалий - нетривиальный вопрос.

Существенные вопросы:
- зависимость качества работы алгоритма от размера случайных подвыборок для каждого дерева
- как влияет выбор случайного базиса на качество работы алгоритма (и почему все спокойно живут с этими полосами)
- выбор оптимального порога скора аномальности для отделения аномалий от нормалий... о, отличное слово, от нормалий, значит, на тестовой выборке.
- доп. вопрос: как связана задача с её решением в одномерном случае независимо для каждого признака?

__________________________________
Видимо, в дальнейшем по дефолту заметки будут делаться по мере прочитывания мной книги Aggarwal, Charu C. (не берусь транлитерировать эту фамилию) "Outlier Analysis". Если там буду какие-то страницы или разделы указывать без пояснений, то это, значит, оттуда.

21.02.17
О постановке задачи. Другим определением аномалии может быть "Есть некоторая нормальная модель, описывающая исходная данные. Аномалия - это то, что плохо вписывается в эту модель". Встречаются случаи, когда все аномалии похожи друг на друга и скапливаются в одной точке. В этом случае предположение, например, о том, что у аномалий относительно далеко находятся ближайшие соседи может не зайти. Также: исходя из такого определения, одним из теоретически оптимальных алгоритмов является применить какой-нибудь моделирующий алгоритм машинного обучения на всех данных в целом и выделить в качестве аномалий объекты с наибольшим отступом.

Также помимо аномалий существует такое понятие, как "шум". Отделять одно от другого задача довольно таки гробовая, однако природа этих вещей разная => и подход к ним тоже может оказаться разный. 
