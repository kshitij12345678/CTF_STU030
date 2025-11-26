import hashlib
import pandas as pd

def compute_student_hash(student_id):
    return hashlib.sha256(student_id.encode()).hexdigest()[:8].upper()

def load_data():
    books = pd.read_csv('books.csv')
    reviews = pd.read_csv('reviews.csv')
    return books, reviews

def filter_books(books):
    return books[(books['rating_number'] == 1234) & (books['average_rating'] == 5.0)]

def find_reviews_with_hash(reviews, hash_str):
    return reviews[reviews['text'].str.contains(hash_str, case=False, na=False)]

def find_target_book(good_books, matching_reviews):
    r_asins1 = set(matching_reviews['parent_asin'].dropna())
    r_asins2 = set(matching_reviews['asin'].dropna()) 
    b_asins = set(good_books['parent_asin'].dropna())
    
    common1 = r_asins1 & b_asins
    common2 = r_asins2 & b_asins
    
    if len(common1) == 1:
        asin = list(common1)[0]
        return good_books[good_books['parent_asin'] == asin].iloc[0]
    elif len(common2) == 1:
        asin = list(common2)[0] 
        return good_books[good_books['parent_asin'] == asin].iloc[0]
    
    return None

def extract_prefix(title, length=8):
    prefix = ""
    for c in title:
        if c != ' ':
            prefix += c
            if len(prefix) == length:
                break
    return prefix

def compute_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()

def save_result(result, filename='flags.txt'):
    with open(filename, 'w') as f:
        f.write(f"RESULT = {result}\n")


student_id = "STU030"
hash_id = compute_student_hash(student_id)
print(f"Student Hash: {hash_id}")

books, reviews = load_data()
print(f"Loaded {len(books)} books and {len(reviews)} reviews")

good_books = filter_books(books)
print(f"Found {len(good_books)} matching books")

matching_reviews = find_reviews_with_hash(reviews, hash_id)
print(f"Found {len(matching_reviews)} reviews with hash")

if len(matching_reviews) > 0:
    target_book = find_target_book(good_books, matching_reviews)
    
    if target_book is not None:
        print(f"Target Book: {target_book['title']}")
        
        title_prefix = extract_prefix(target_book['title'])
        result_hash = compute_hash(title_prefix)
        
        print(f"Title Prefix: {title_prefix}")
        print(f"Result Hash: {result_hash}")
        
        save_result(result_hash)
        print("Result saved to output.txt")
    else:
        print("No target book found")
else:
    print("No matching reviews found")
