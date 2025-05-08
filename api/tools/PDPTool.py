from langchain_core.tools import tool
import requests
from lxml import html

@tool
def pdp_news_scraper(page: int = None) -> list:
    """
    Scrapes PDP news from BloombergHT for one or multiple pages.
    
    Parameters:
        - page: Specific page number to scrape. If None, scrapes the last 5 pages by default.

    Returns:
        - A list of dictionaries with 'url' and 'content' keys
    """
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    results = []

    try:
        pages_to_scrape = [page] if page else range(1, 6)

        for p in pages_to_scrape:
            url = f"https://www.bloomberght.com/borsa/hisseler/kap-haberleri/{p}"
            response = requests.get(url, headers=headers)
            tree = html.fromstring(response.content)

            for i in range(2, 8):
                link = tree.xpath(f'/html/body/main/div[1]/div[2]/div[3]/div/div/div[{i}]/a/@href')
                if not link:
                    continue

                full_url = f"https://www.bloomberght.com{link[0]}"
                kap_response = requests.get(full_url, headers=headers)
                kap_tree = html.fromstring(kap_response.content)
                texts = kap_tree.xpath('/html/body/main/div[1]/div[2]/div/article/div[2]/div/table/tbody//text()')
                cleaned_text = ' '.join(t.strip() for t in texts if t.strip())

                results.append({
                    "url": full_url,
                    "content": cleaned_text if cleaned_text else "❗ İçerik bulunamadı"
                })

        return results if results else [{"error": "No news items found."}]

    except Exception as e:
        return [{"error": str(e)}]
