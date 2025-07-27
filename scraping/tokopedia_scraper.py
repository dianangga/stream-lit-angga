# scraper.py

import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


# def get_driver():
#     options = Options()
#     options.add_argument("--headless")  # Bisa dihapus untuk debug
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
#     service = Service(ChromeDriverManager().install())
#     return webdriver.Chrome(service=service, options=options)

# def open_url(driver, url):
#     try:
#         driver.get(url)
#         time.sleep(5)
#         return True
#     except Exception as e:
#         print(f"Error opening URL: {e}")
#         return False

# # def scroll_to_reviews(driver):
#     """Scroll down to the reviews section"""
#     try:
#         # Look for review section with more specific selectors
#         review_selectors = [
#             "//div[@id='review-feed']",
#             "//div[contains(@data-testid, 'review')]",
#             "//div[contains(text(), 'Ulasan')]",
#             "//div[contains(text(), 'Review')]",
#             "//section[contains(@class, 'review')]",
#             "//*[@id='review-feed']/article[1]/div/p[2]/span",
#             "/html/body/div[1]/div/main/div[2]/div[1]/div/section/section/article[1]/div/p[2]/span",
#         ]
        
#         review_section = None
#         for selector in review_selectors:
#             try:
#                 review_section = WebDriverWait(driver, 5).until(
#                     EC.presence_of_element_located((By.XPATH, selector))
#                 )
#                 break
#             except:
#                 continue
        
#         if review_section:
#             driver.execute_script("arguments[0].scrollIntoView();", review_section)
#             time.sleep(2)
#             print("Successfully scrolled to review section")
#             return True
#         else:
#             print("Review section not found, scrolling to bottom")
#             driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#             time.sleep(3)
#             return False
            
#     except Exception as e:
#         print(f"Could not find review section: {e}")
#         driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
#         time.sleep(3)
#         return False

# # def load_more_reviews(driver):
#     """Click 'Load More' or 'Show More' buttons to load additional reviews"""
#     try:
#         # More comprehensive selectors for load more buttons
#         load_more_selectors = [
#             "//button[contains(text(), 'Lihat Semua')]",
#             "//button[contains(text(), 'Tampilkan Lebih')]",
#             "//button[contains(text(), 'Load More')]",
#             "//button[contains(@data-testid, 'load-more')]",
#             "//a[contains(text(), 'Lihat Semua')]",
#             "//*[@id='review-feed']/article[10]/div/p[2]/button",
#             "//button[contains(text(), 'Selengkapnya')]",
#             "//button[contains(@class, 'load-more')]",
#             "//div[@id='review-feed']//button[contains(text(), 'Lihat')]",
#         ]
        
#         for selector in load_more_selectors:
#             try:
#                 load_more_btn = WebDriverWait(driver, 3).until(
#                     EC.element_to_be_clickable((By.XPATH, selector))
#                 )
#                 # Scroll to button first
#                 driver.execute_script("arguments[0].scrollIntoView();", load_more_btn)
#                 time.sleep(1)
#                 # Click using JavaScript to avoid interception
#                 driver.execute_script("arguments[0].click();", load_more_btn)
#                 time.sleep(3)
#                 print(f"Clicked load more button with selector: {selector}")
#                 return True
#             except:
#                 continue
        
#         return False
#     except Exception as e:
#         print(f"Error loading more reviews: {e}")
#         return False

# # def click_next_page(driver):
#     """Click next page button for pagination"""
#     try:
#         next_page_selectors = [
#             "//button[contains(@aria-label, 'Next page')]",
#             "//button[contains(@aria-label, 'Halaman berikutnya')]",
#             "//a[contains(@aria-label, 'Next page')]",
#             "//button[contains(text(), 'Next')]",
#             "//button[contains(text(), 'Selanjutnya')]",
#             "//button[@data-testid='btnShopReviewNext']",
#             "//div[contains(@class, 'pagination')]//button[not(@disabled)]//following-sibling::button",
#             "//*[@id='zeus-root']/div/main/div[2]/div[1]/div/section/div[3]/nav/ul/li[11]/button",
#             "//nav//ul//li[position()=last()]//button",
#             "//nav//ul//li[position()=last()-1]//button"
#         ]
        
#         for selector in next_page_selectors:
#             try:
#                 next_button = WebDriverWait(driver, 5).until(
#                     EC.element_to_be_clickable((By.XPATH, selector))
#                 )
#                 # Check if button is enabled
#                 if next_button.is_enabled() and next_button.get_attribute('disabled') is None:
#                     driver.execute_script("arguments[0].scrollIntoView();", next_button)
#                     time.sleep(1)
#                     driver.execute_script("arguments[0].click();", next_button)
#                     time.sleep(5)  # Wait for page to load
#                     print(f"Clicked next page button with selector: {selector}")
#                     return True
#             except:
#                 continue
        
#         print("No next page button found or all buttons are disabled")
#         return False
#     except Exception as e:
#         print(f"Error clicking next page: {e}")
#         return False

# def scrape_comments_from_page(driver):
#     """Scrape comments from current page"""
#     comments = []
    
#     # Primary selector based on your finding
#     primary_selector = "//div[@id='review-feed']//article//span[string-length(text()) > 5]"

    
#     # Fallback selectors in case the structure changes
#     fallback_selectors = [
#         "//*[@id='review-feed']//article//p//span[string-length(text()) > 5]",
#         "//*[@id='review-feed']//article//span[contains(@class, 'review') or contains(@data-testid, 'review')]",
#         "//*[@id='review-feed']//article//div//span[string-length(text()) > 5]",
#         "//div[contains(@data-testid, 'review-content')]",
#         "//span[@data-testid='lblItemUlasan']",
#         "//*[@id='review-feed']//article//p[2]//span",
#         "//*[@id='review-feed']//article//div//p//span",
#         "//*[@id='review-feed']/article[1]/div/p[2]/span"
#     ]
    
#     # Try primary selector first
#     try:
#         comment_elements = driver.find_elements(By.XPATH, primary_selector)
#         if comment_elements:
#             print(f"Found {len(comment_elements)} comments with primary selector")
#             for element in comment_elements:
#                 try:
#                     comment_text = element.text.strip()
#                     if comment_text and len(comment_text) > 5:  # Filter out very short texts
#                         comments.append(comment_text)
#                 except:
#                     continue
#         else:
#             print("No comments found with primary selector, trying fallback selectors...")
            
#             # Try fallback selectors
#             for selector in fallback_selectors:
#                 try:
#                     comment_elements = driver.find_elements(By.XPATH, selector)
#                     if comment_elements:
#                         print(f"Found {len(comment_elements)} comments with fallback selector: {selector}")
#                         for element in comment_elements:
#                             try:
#                                 comment_text = element.text.strip()
#                                 if comment_text and len(comment_text) > 5:
#                                     comments.append(comment_text)
#                             except:
#                                 continue
#                         break
#                 except Exception as e:
#                     print(f"Error with fallback selector {selector}: {e}")
#                     continue
                    
#     except Exception as e:
#         print(f"Error with primary selector: {e}")
    
#     return comments

# # def scrape_all_comments(driver):
#     """Scrape comments from all pages"""
#     all_comments = []
#     page_number = 1
#     max_pages = 3  # Safety limit
    
#     # Scroll to reviews section
#     scroll_to_reviews(driver)
    
#     while page_number <= max_pages:
#         print(f"\n--- Scraping Page {page_number} ---")
        
#         # Try to load more reviews on current page
#         for i in range(3):  # Try loading more reviews 3 times
#             if load_more_reviews(driver):
#                 time.sleep(2)  # Wait for new reviews to load
#             else:
#                 break
        
#         # Scrape comments from current page
#         page_comments = scrape_comments_from_page(driver)
        
#         if page_comments:
#             all_comments.extend(page_comments)
#             print(f"Page {page_number}: Found {len(page_comments)} comments")
#         else:
#             print(f"Page {page_number}: No comments found")
        
#         # Try to go to next page
#         if click_next_page(driver):
#             page_number += 1
#             time.sleep(3)  # Wait for page to fully load
            
#             # Scroll to reviews section on new page
#             scroll_to_reviews(driver)
#         else:
#             print(f"No more pages available. Stopped at page {page_number}")
#             break
    
#     # Remove duplicates while preserving order
#     unique_comments = []
#     seen = set()
#     for comment in all_comments:
#         if comment not in seen:
#             unique_comments.append(comment)
#             seen.add(comment)
    
#     print(f"\nTotal comments collected: {len(all_comments)}")
#     print(f"Unique comments after deduplication: {len(unique_comments)}")
    
#     return unique_comments

# def save_cookies(driver, filename):
#     """Save cookies to file"""
#     try:
#         cookies = driver.get_cookies()
#         with open(filename, 'w') as f:
#             json.dump(cookies, f)
#         print(f"Cookies saved to {filename}")
#     except Exception as e:
#         print(f"Error saving cookies: {e}")

# def load_cookies(driver, filename):
#     """Load cookies from file"""
#     try:
#         with open(filename, 'r') as f:
#             cookies = json.load(f)
#         for cookie in cookies:
#             try:
#                 driver.add_cookie(cookie)
#             except:
#                 pass  # Skip invalid cookies
#         print(f"Cookies loaded from {filename}")
#         return True
#     except Exception as e:
#         print(f"Error loading cookies: {e}")
#         return False
# scraper_refactored.py

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
import time

# def get_driver():
#     return webdriver.Chrome(service=Service(ChromeDriverManager().install()))

def get_driver():
    options = Options()
    options.add_argument("--headless")  # Bisa dihapus untuk debug
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=options)

def open_url(driver, url):
    try:
        driver.get(url)
        time.sleep(5)
        return True
    except Exception as e:
        print(f"Error opening URL: {e}")
        return False

def scrape_tokopedia_reviews(driver):
    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'css-dfpqc0'))
        )
        print("Klik tombol escape")
        body = driver.find_element(By.TAG_NAME, 'body')
        body.send_keys(Keys.ESCAPE)
        driver.execute_script("window.scrollBy(0, 2000);")
        driver.execute_script("window.scrollBy(0, 2000);")
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, 'css-15m2bcr'))
        )

        isFound = True
        all_reviews = []

        while isFound:
            # Tekan semua tombol "Selengkapnya" yang ada di halaman
            selengkapnya_buttons = driver.find_elements(By.CLASS_NAME, "css-89c2tx")
            for btn in selengkapnya_buttons:
                try:
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(0.3)  # Delay kecil untuk memastikan isi dimuat
                except Exception as e:
                    print("Gagal klik tombol Selengkapnya:", e)

            # Ambil ulang page_source setelah tombol diklik
            soup = BeautifulSoup(driver.page_source, "html.parser")
            containers = soup.find_all('article', {'class': 'css-15m2bcr'})
            ulasans = []

            for c in containers:
                ulasan = c.find('span', {'data-testid': 'lblItemUlasan'})
                if ulasan and ulasan.text.strip():
                    ulasans.append(ulasan.text.strip())
                    print(ulasan.text)
                    print("=" * 40)

            all_reviews.extend(ulasans)
            print(f"\nDitemukan {len(ulasans)} ulasan di halaman ini\n")

            # Coba klik tombol "Laman berikutnya"
            try:
                next_button = driver.find_element(By.XPATH, '//button[@aria-label="Laman berikutnya"]')
                if next_button.get_attribute("disabled") is not None:
                    print("Tombol next tidak aktif, berhenti.")
                    isFound = False
                else:
                    driver.execute_script("arguments[0].click();", next_button)
                    print("Klik tombol next...")
                    time.sleep(3)
            except Exception:
                print("Tidak ditemukan tombol next. Berhenti.")
                isFound = False

        driver.quit()
        return all_reviews

    except Exception as e:
        print("Terjadi error saat mengambil data:", e)
        driver.quit()
        return []

    
def save_cookies(driver, filename):
    """Save cookies to file"""
    try:
        cookies = driver.get_cookies()
        with open(filename, 'w') as f:
            json.dump(cookies, f)
        print(f"Cookies saved to {filename}")
    except Exception as e:
        print(f"Error saving cookies: {e}")

def load_cookies(driver, filename):
    """Load cookies from file"""
    try:
        with open(filename, 'r') as f:
            cookies = json.load(f)
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except:
                pass  # Skip invalid cookies
        print(f"Cookies loaded from {filename}")
        return True
    except Exception as e:
        print(f"Error loading cookies: {e}")
        return False