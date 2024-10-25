
## Kezdet
Az egész úgy indult, hogy évekig a Bitcoin érdekelt, de az ottani barátaim egyre többet beszéltek az AI-ról. __Be is léptem egy csoportba, ahol az volt a fő téma__.
Amikor láttam, hogy nem bírok a bejövő infókkal, akkor feltettem a kérdést egy nálam profibbnak, hogy mivel kezdjek.. Ő a __Pytorch Lightning könyvet javasolta__.

## Ösvény
Szeretek belemélyedni a dolgokba, amikor első olvasásra láttam, hogy nem értem meg a könyvet, akkor úgy határoztam, hogy teljesen lefordítom magyarra.
Azóta számos feladatot úgy jelöltem ki magamnak, hogy a könyvben szerepelt, és továbbfejlesztettem. Legkomolyabb vállalásom a BTC árfolyampredikciója volt, Time Series modellel.
Tisztában voltam vele, hogy nem működhet, mivel az adatkészletem időtartama alatt az alapvető környezet változott, és se ciklikusság, se a külső behatásoktól mentesség nem igaz rá.
Viszont a pandas és vizualizációs eszközök fontosságát tanultam meg ettől a projekttől. És még pár furcsa hipotézis vert gyökeret a fejemben.
Utána arra indultam, ami a könyvből kimaradt, alap folyamatok mélyebb megértése. Igazából ez jelenleg is tart, ha alapvető igazságként hivatkoznak valamire, amit nem értek, akkor próbálok gazdagon animált videót megnézni a témáról.

Időközben szembejött a __Stable Baselines 3__, és mivel mindig vonzott az, amit nem kötnek az orrunkra, ezért igyekeztem megérteni – gyakorlatban ehhez is annyi helpet fordítottam, amit csak praktikusnak láttam.
Az SB3 egyébként ideális vasúthoz is, próbáltam is felkészülni a Hackathonra belőle, de sajnos egy hetet hagytam egy hónapos munkára. A layerek felépítését ebben rekurzív függvénnyel gondoltam megoldani, de annyira utolsó pillanatban lett kész csak a függvény, egy bizonytalan környezetben, hogy jobbnak láttam, ha nem erőltetem a versenyen, és sima ismételt for ciklusokkal dolgozunk.
Az SB3 azért jó, mert az aktor lehet például egy szerelvény, és az observation space a teljes elérhető vágányrendszer, az action space pedig a következő elérhető állapotváltozás a következő pályablokk állapota.
Mivel nincs ilyen Gym az SB3-hoz, ezért átmenetileg suspendeltem a feladatot.

Úgy véltem, más, egyszerűbb feladattal kéne átmenetileg foglalkoznom, mint például a soron következő példa a könyvben. Ellenben ez egy stackelt modellkombó, ahol a kapott képből alkot mondatokat. Mivel ez nemcsak képfeldolgozót hanem nyelvi modell toolkitet is tartalmaz, ami valíszínű verzióváltások miatt már nem működik, szokás szerint nagyobb munkát igényel, mint gondoltam. De mivel később én is modellek kombinációját akarom használni, ezen a vonalon akarok végigmenni.

## Eddigi modelljeim:
https://github.com/vmistvan/MIA_helyzet/tree/main
XOR logikai kaput megvalósító modell – persze ki kellett próbálnom, hogy hogyan működik, ha továbbfejlesztem szorzótáblának..
Bert modell – és privátban ez persze már magyarul is osztályoz.
Timeseries – persze nyilvánosan cask a forgalmi adatokat megjósoló változat, mert a BTC olyan helyzetet okozhatna, amit nem akarok magamra venni.
Van egy beszédfelismerő is ami magyarul is tud, de ezt nem másoltam bele a fenti repóba, külön található meg, kaptam hibajegyet is, egy ismeretlentől 😊
Food_GAN – a kaja okés, de nyilván nem érti a mélyebb 3d geometriát, engem nagyon zavarnak az amorf generált tányérok
Semisupervised -  A legutolsó projekt, feltettem lángbetűs figyelmeztetéssel, hogy nem jó még, de hátha valaki így is hasznát leli. Meg stay tuned.

## Részvétel:
Előadás a cégemnél az AI-ról (200 hallgatóval!).
Folyamatos párbeszéd az online csoportomban, és emberekkel külön is, beszámoló az eredményeimről és vélemény kikérése.
Könyv lefordítása, és más csak angol területen elérhető háttérinfók fordítása. Szerzői jogi problémák miatt ezek nem publikusak (a bennem élő hörcsög megnyugtatására alcímmel 😃 )

