# Projektna dokumentacija: LUMEN2026

## 1. Sažetak projekta

LUMEN2026 je projekt za analizu kupaca i preporuku proizvoda na temelju povijesti kupnji. Projekt kombinira klasični pristup s inženjeringom značajki, redukcijom dimenzionalnosti, klasteriranjem i graf-based rangiranjem proizvoda.

U praksi postoje dva povezana smjera rada:

- klasični recommendation pipeline u mapi `classic_methods/`
- eksperimentalna analiza klastera kupaca u notebookima `vae_customer_clustering.ipynb` i `clustering.ipynb`

## 2. Podaci i filtriranje

Ulazni skup podataka je datoteka `LUMEN_DS.csv`. Prije modeliranja provodi se filtriranje kupaca s premalo povijesti kupnje. U notebooku `vae_customer_clustering.ipynb` koristi se prag od:

- najmanje 50 kupnji po kupcu
- najmanje 5 različitih artikala po kupcu

Ovo smanjuje šum i uklanja kupce s prekratkom poviješću koji bi mogli destabilizirati embedding i klastere.

## 3. Izrada široke tablice značajki

Nakon filtriranja podataka gradi se široka customer-level tablica značajki. Svaki redak predstavlja jednog kupca, a stupci sažimaju ponašanje kroz agregate poput:

- broj kupnji i broj jedinstvenih artikala
- ukupna i prosječna vrijednost narudžbi
- marža i omjeri isporuke
- entropija kupovnih obrazaca
- recency i tenure
- udjeli po grupama proizvoda i familijama proizvoda

Glavna funkcija za to je `build_customer_feature_matrix()` u `classic_methods/src/embeddings/customer_features.py`.

## 4. VAE embedding

Široka tablica značajki se zatim standardizira i ulazi u tabular VAE model. VAE komprimira korisnike u manji latentni prostor. U notebooku se koristi latentna dimenzija 16.

Važno je napomenuti da izlaz dekodera nije klasični statistički z-score test, nego vrijednost u standardiziranom feature spaceu. Ako je dekodirana vrijednost oko 1.5, to znači da je taj centroid oko 1.5 standardne devijacije iznad prosjeka za tu značajku.

## 5. KMeans klasteriranje

Na latentnim vektorima radi se KMeans klasteriranje. U notebooku `vae_customer_clustering.ipynb` prikazuje se elbow chart kako bi se procijenio broj klastera.

Broj klastera je namjerno postavljen preko varijable `CHOSEN_K`, tako da se lako može promijeniti bez preuređivanja notebooka.

## 6. Dekodiranje centroida i interpretacija

Nakon klasteriranja uzimaju se KMeans centroidi u latentnom prostoru i prolaze kroz VAE decoder. Tako se dobiva reprezentativni vektor svakog klastera u standardiziranom feature spaceu.

Interpretacija klastera temelji se na značajkama koje su najjače pozitivno ili negativno odstupale od prosjeka. Zato se klasteri opisuju kao ponašajni profili, primjerice:

- kupci s visokom aktivnošću i dugom poviješću
- kupci s većom potrošnjom i maržom
- kupci sa širim asortimanom kupnje
- kupci s koncentriranom ili specijaliziranom kupnjom

Takva interpretacija je heuristička, ali korisna za poslovno razumijevanje segmenata.

## 7. Usporedba dva klasteriranja

Osim VAE klasteriranja, u projektu postoji i klasično KMeans klasteriranje iz notebooka `clustering.ipynb`. Tamo kupci prvo prolaze kroz jednostavniji pristup temeljen na udjelima grupa proizvoda.

Za usporedbu se može napraviti matrica 4x4 gdje redci predstavljaju klastere iz `vae_customer_clustering.ipynb`, a stupci klastere iz `clustering.ipynb`. Svaka ćelija pokazuje koliki postotak kupaca iz VAE klastera završava u određenom klasičnom klasteru.

## 8. Kako pokrenuti projekt

1. Aktivirati virtualno okruženje iz korijena repozitorija.
2. Instalirati zavisnosti za `classic_methods`.
3. Pokrenuti notebookove `vae_customer_clustering.ipynb` i `clustering.ipynb`.
4. Po potrebi promijeniti `CHOSEN_K` ili pragove filtriranja.

Primjer:

```bash
source .venv/bin/activate
pip install -e classic_methods
```

## 9. Zaključak

Projekt je prvenstveno preporučivački sustav za kupce i artikle. Klasični pipeline koristi široku tablicu značajki, latentni VAE embedding i Katz rangiranje, dok notebookovi služe za interpretaciju segmenata kupaca i usporedbu različitih načina klasteriranja.
