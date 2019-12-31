# GraphologyWithAnn
Yapay Sinir Ağları ile El Yazısı Harflerinden Karakter Analizi

Literatürde Grafoloji bilimi olarak geçen; el yazısından, kişinin karakter analizinin yapıldığı bilim dalının yapay zeka ile küçük bir bölümünün uygulanması.

Bu projede karakter analizi sadece 4 harfin yazılışına bakılarak 3 farklı analiz şeklinde ele alınmıştır. Bu harfler g,y,h ve t harfleridir. Bu harflerden yapılan analizler ise; kişinin iletişim becerisinin düşük veya yüksek olması, enerjik veya sabırsız olması ve hırslı olup olmaması üzerine yapılan analizlerdir.

Öncelikle bir veya 2 katmanlı olacak şekilde(iki şekilde de eğitilebilir) bir yapay sinir ağı oluşturuldu. Bu yapay sinir ağı kullanılırken matematiksel işlemler için kullanılan numpy kütüphanesi dışında hiç bir kütüphane kullanılmadı.

Kaggle üzerinden bir çok farklı kişi tarafından yazılmmış olan el yazılarının resim olarak bulunduğu veri seti kullanıldı. Bu veriler içerisinden değişik özelliklere sahip 200'den fazla veri seçilip bir resim veri seti oluşturuldu. Bu resimler sayısallaştırılıp bir CSV dosyası içine konularak veri seti oluşturuldu. Daha sonra bu verilere OpenCV kütüphanesi aracılığıyla Binarization(ikilileştirme) uygulandı. Resimler artık siyah veya beyaz(0 veya 1) şeklinde tutuldu. 

Bu veri seti aracılığıyla oluşturulmuş olan yapay sinir ağı eğitildi. Eğitim sonucu matplolib kütüphanesi ile grafiğe döküldü. Eğitim tamamlandıktan sonra ister test için ayrılmış olan veriler arasından ister kendi el yazınız üzerinden sistem test edilip, kişilik analizi yapılabilir.

Proje bit web projesi olarak tasarlandı ve bu işlemler için Flask kullanıldı. Flask aracılığıyla web ortamında oluşturuldu. 
