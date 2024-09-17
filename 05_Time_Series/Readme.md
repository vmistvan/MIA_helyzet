# Time Series - Idősor modellek

## Kedves ellenségem ez a modelltípus :) Ne várj ettől a projekttől túl sokat!

Valójában sokkal többet dolgoztam ezzel a modelltípussal, és minden probléma megoldása újabb kérdéseket vetett fel, amiket aztán szintén megoldva egyre inkább elmaradt a modell az elvárásaimtól. Valójában nem is számítottam jobbra. Viszont ebben van minden, amiből lehet tanulni!
Kell tisztítani az adatokat, és a pandas szuper tulajdonságait tudod élesben kipróbálni, szóval adattudóskodhatsz, elmerülhetsz a vizualizációban, van benne __feature engineering__, és predikció. Ja, igen. Ez prediktív és nem regressziós AI. Valójában a legfőbb oka a sikertelenségemnek az volt, hogy próbáltam árfolyamadatokra átírni a modellt, egy nehéz időszakra, amikor
1. folyamatos előre nem látható - azaz nem prediktálható beavatkozás volt az árfolyamba,
2. semmi ciklikussága nem volt az árfolyamadatoknak, és még visszatekintve sem tudok trendet meghatározni.

Ennek ellenére jó volt a hiperparamétereket tekergetni, és nézni, hogy ad vissza jobb eredményt, mekkora időablakokat használjak.

Ha elfogadsz egy tanácsot, akkor ezerszer inkább hőmérsékletadatokra írod át! :)

Valójában ez egy példa hibáinak kijavítása, ezért nincs benne explicit predikció, egyszer talán beledolgozom azt is a példaprogramba, amit én működőnek találtam, vagy megosztom azt a programom, amivel a Bitcoin árfolyamgörbéjét szálaztam. elvesztette titkos jellegét, hisz pont nem lehetett semmire használni :)

Ide is fog egy __install.txt__ kerülni, a futtatáshoz szükséges tudnivalókkal. Gépigénye elenyésző, kisfloppyra (tudod még mi az?) ráférne a modell :)

## Eredeti leírásom:

# Time series alap modell

Az idősor modellek példaprogramja ez.
A modell fut és konvergál.

Fontos tanulság az alábbi sor, amit nem tartalmazott az eredeti példa, de ha el akarjuk kerülni, hogy az adataink cpu és cuda között szétszóródjanak, akkor hozzá kell adni a kódhoz:
<pre>torch.set_default_tensor_type('torch.cuda.FloatTensor')</pre>

## Mit tanulhatunk a példából?
- A pandas python könyvtárral nagyon jól lehet használni a csv fájlokat, és adatzsonglőrködhetünk. A példa sok hasznos függvényt használ.
- A pandas nemcsak egy tömböt készít a beolvasott csv fájlból, hanem egy objektumot. A példa nagyon sok olyan esetet mutat meg, amikor egy előre megírt, nagyszerűen használható funkciót vethetünk be az adatok kezelésére.
- A sklearn fejlett, AI támogató függvényekkel szintén nagyon hasznos.
- Alapból kikommentezve megtalálható egy grafikonkészítő rutin, ami a vizualizációhoz igencsak hasznos tud lenni.
- Az alap AI funkciók megértéséhez ugyan érdemes egyszerűbb példát megnézni, de ebben a példában a neurális háló felépítését és életciklus metódusok készítését is láthatjuk. Valóban egy kraftolt, részletesen beállított példa a modellek fejlesztéséhez
- Sok a komment, amik sokszor megmutatják azt a küzdelmet, hogy időközben elavult függvények, vagy talán soha nem működött kódsoroknál hogy kerestem a hibát...
- Lehet, hogy 100 találat a neten ugyanazt a használhatatlan javaslatot fogja adni, de a keresést kicsit finomítva, esélyes, hogy a 101-edik válasz használható lesz
- Sokszor amikor hibát dob, a teljes sort belemásolom a keresőbe. A modern keresőmotorok figyelmen kívül tudják hagyni a változónevek, stb miatt érdektelen szövegrészeket, és megfelelő találatokat adnak. Ezt egészítem ki az általam feltett kérdéssel.
