const gplay = require('google-play-scraper').memoized();
const fs = require('fs');


const terms = fs.readFileSync('terms.txt','utf8').split('\n')

async function getApps(term) {
    return gplay.search({
        term: term,
        num: 100,
        fullDetail: true,
    }).catch((err) => {
        console.log(term)
        console.log(err)
        return [];
    })
}

async function main() {
    let writeStream = fs.createWriteStream("data.csv");
    writeStream.write(['id', 'kategoria', 'srednia_ocena', 'liczba_opinii', 'liczba_ocen', 'rozmiar_aplikacji',
        'ilosc_instalacji', 'grupa_docelowa', 'cena', 'ostatnia_aktualizacja', 'wspierany_android',
        'reklamy', 'polecana', 'dlugosc_opisu', 'dlugosc_podsumowania'].join(',') + '\n')
    for (const term of terms) {
        const data = await getApps(term);
        for (const app of data) {
            writeStream.write([app.appId, app.genreId, app.score, app.reviews, app.ratings, app.size, app.minInstalls,
                app.contentRating, app.price, app.updated, app.androidVersion,
                app.adSupported, app.editorsChoice, app.description.length, app.summary.length].join(',') + '\n');
        }
    }
}

main()


