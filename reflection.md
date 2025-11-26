# Reflection on CTF Challenge

This CTF challenge was quite interesting and require different approach for each flag. Here is my experience solving it.

For FLAG1, I start by computing hash from my student ID "STU030" using SHA256 and taking first 8 characters. Then I filter books dataset to find books with rating_number=1234 and average_rating=5.0, which give me 150 books. Next step was finding reviews that contain my hash "44AA097D" - only found 1 review with text "44AA097D outstanding amazing experience". I match this review to book using ASIN and found the manipulated book "Pretty Honest: The Straight-Talking Beauty Companion". Finally extract first 8 non-space chars "PrettyHo" from title and hash it for FLAG1.

FLAG2 was more straightforward - the fake review is the one containing my hash, so FLAG2 is simply "FLAG2{44AA097D}".

FLAG3 was most challenging part involving machine learning. I create heuristic labels for suspicious vs genuine reviews based on rating=5, text length, and word content. Suspicious reviews are short with superlatives like "amazing", "best". Genuine reviews are long with domain-specific words like "characters", "plot", "narrative". Since target book had only 11 reviews, I add synthetic examples to train classifier properly. Used TfidfVectorizer and LogisticRegression, then SHAP analysis to find words that reduce suspicion scores. Top 3 words were "story", "development", "narrative" which make sense as genuine book review language. Final FLAG3 concatenate these words with numeric ID "030" and hash first 10 characters.

The challenge combine data filtering, hash-based detection, and ML interpretability nicely.