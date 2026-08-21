from langchain_core.tools import tool
import requests
from lxml import html

# Each KAP-news detail page has a heading formatted "TICKER/Full Company
# Name" (confirmed live, e.g. "BALAT/BALATACILAR BALATACILIK SANAYI VE
# TICARET A.S."), which is what lets us match a disclosure to the ticker
# that was actually asked about instead of trusting the LLM to notice a
# mismatch in an unfiltered general feed.
DETAIL_TICKER_XPATH = '/html/body/main/div[1]/div[2]/div/article/div[1]/div[1]/h1/text()'
DETAIL_CONTENT_XPATH = '/html/body/main/div[1]/div[2]/div/article/div[2]/div/table/tbody//text()'


def _normalize_ticker(ticker: str) -> str:
    return ticker.upper().split(".")[0].strip()


@tool
def pdp_news_scraper(ticker: str, pages_to_scan: int = 5) -> list:
    """
    Scrapes recent PDP (KAP) disclosure summaries via BloombergHT's public
    KAP-news aggregation pages (not KAP's own site directly), filtered to
    disclosures that actually belong to the given ticker - each candidate
    disclosure's own "TICKER/Company Name" heading is checked against the
    requested ticker before it's included, so an unrelated company's
    disclosure is never returned as if it were about the requested one.

    Parameters:
        - ticker: Stock ticker symbol (e.g. 'THYAO.IS'). Only disclosures
          for this company are returned.
        - pages_to_scan: how many recent KAP-news listing pages to scan
          looking for a match (default 5).

    Returns:
        - A list of dicts with 'url' and 'content' keys for matching
          disclosures. If none are found in the scanned pages, returns a
          single dict {"info": "..."} stating that explicitly instead of
          any unrelated disclosure.
    """
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    target = _normalize_ticker(ticker)
    results = []

    try:
        for p in range(1, pages_to_scan + 1):
            url = f"https://www.bloomberght.com/borsa/hisseler/kap-haberleri/{p}"
            response = requests.get(url, headers=headers, timeout=10)
            tree = html.fromstring(response.content)

            for i in range(2, 8):
                link = tree.xpath(f'/html/body/main/div[1]/div[2]/div[3]/div/div/div[{i}]/a/@href')
                if not link:
                    continue

                full_url = f"https://www.bloomberght.com{link[0]}"
                kap_response = requests.get(full_url, headers=headers, timeout=10)
                kap_tree = html.fromstring(kap_response.content)

                heading = kap_tree.xpath(DETAIL_TICKER_XPATH)
                disclosure_ticker = heading[0].split("/")[0].strip().upper() if heading else None
                if disclosure_ticker != target:
                    continue

                texts = kap_tree.xpath(DETAIL_CONTENT_XPATH)
                cleaned_text = ' '.join(t.strip() for t in texts if t.strip())

                results.append({
                    "url": full_url,
                    "content": cleaned_text if cleaned_text else "❗ İçerik bulunamadı"
                })

        if not results:
            return [{"info": f"No recent KAP/PDP disclosures found for {ticker} in the last {pages_to_scan} feed page(s)."}]
        return results

    except Exception as e:
        return [{"error": str(e)}]
