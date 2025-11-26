# CTF Challenge STU030

This repository contain solution for the CTF challenge where we need find manipulated book and fake reviews using data analysis and machine learning techniques.

## Approach Summary

The challenge have 3 main parts:

1. **Find Manipulated Book (FLAG1)**
   - Compute hash from student ID "STU030" 
   - Search for reviews containing this hash
   - Match reviews with books having rating_number=1234 and average_rating=5.0
   - Extract first 8 non-space characters from book title and hash it

2. **Identify Fake Review (FLAG2)**
   - The fake review is one that contain our hash ID
   - Format: FLAG2{HASH_ID}

3. **Machine Learning Analysis (FLAG3)**
   - Filter reviews for target book only
   - Create labels: suspicious vs genuine reviews
   - Train text classifier using TfidfVectorizer and LogisticRegression
   - Use SHAP analysis to find words that reduce suspicion
   - Take top 3 words + numeric ID and hash first 10 characters

## Files Structure

- `solver.py` - Main solution code with all analysis
- `flags.txt` - Contains all three flags
- `reflection.md` - Detailed explanation of approach
- `.gitignore` - Exclude CSV data files

## Usage

```bash
python3 solver.py
```

Make sure you have the CSV files (books.csv, reviews.csv) in same directory.

## Dependencies

- pandas
- scikit-learn  
- shap
- numpy
- hashlib (built-in)