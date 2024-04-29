get_diet_recommendation = {
    'Fungal infection': ['Garlic', 'Yogurt', 'Turmeric', 'Neem', 'Coconut Oil'],
    'Allergy': ['Turmeric', 'Honey', 'Green Tea', 'Ginger', 'Omega-3 Rich Foods'],
    'GERD': ['Oatmeal', 'Ginger Tea', 'Leafy Greens', 'Banana', 'Almond Milk'],
    'Chronic cholestasis': ['Beetroot', 'Carrots', 'Papaya', 'Spinach', 'Turmeric'],
    'Drug Reaction': ['Aloe Vera', 'Coconut Water', 'Mint Leaves', 'Cucumber', 'Banana'],
    'Peptic ulcer disease': ['Cabbage Juice', 'Bananas', 'Oatmeal', 'Yogurt', 'Honey'],
    'AIDS': ['Protein-Rich Foods', 'Fruits', 'Vegetables', 'Whole Grains', 'Nuts and Seeds'],
    'Diabetes': ['Bitter Gourd', 'Fenugreek Seeds', 'Indian Gooseberry', 'Turmeric', 'Cinnamon'],
    'Gastroenteritis': ['Banana', 'Boiled Potatoes', 'Yogurt', 'Ginger Tea', 'White Rice'],
    'Bronchial Asthma': ['Garlic', 'Turmeric', 'Ginger', 'Mustard Oil', 'Honey'],
    'Hypertension': ['Banana', 'Leafy Greens', 'Oats', 'Berries', 'Garlic'],
    'Migraine': ['Ginger Tea', 'Peppermint Tea', 'Dark Chocolate', 'Salmon', 'Almonds'],
    'Cervical spondylosis': ['Carrot Juice', 'Turmeric', 'Garlic', 'Leafy Greens', 'Ginger Tea'],
    'Paralysis (brain hemorrhage)': ['Blueberries', 'Walnuts', 'Salmon', 'Spinach', 'Green Tea'],
    'Jaundice': ['Lemon Juice', 'Papaya', 'Bottle Gourd Juice', 'Beetroot', 'Radish'],
    'Malaria': ['Cinnamon Tea', 'Ginger Tea', 'Orange Juice', 'Papaya', 'Coconut Water'],
    'Chicken pox': ['Oatmeal Bath', 'Neem Leaves', 'Baking Soda', 'Green Tea', 'Honey'],
    'Dengue': ['Papaya Leaf Juice', 'Coconut Water', 'Broth Soups', 'Porridge', 'Herbal Tea'],
    'Typhoid': ['Banana', 'Plain Rice', 'Applesauce', 'Boiled Potatoes', 'Ginger Tea'],
    'Hepatitis A': ['Turmeric', 'Garlic', 'Leafy Greens', 'Beetroot', 'Carrot Juice'],
    'Hepatitis B': ['Garlic', 'Leafy Greens', 'Carrot Juice', 'Turmeric', 'Ginger Tea'],
    'Hepatitis C': ['Turmeric', 'Garlic', 'Leafy Greens', 'Beetroot', 'Carrot Juice'],
    'Hepatitis D': ['Garlic', 'Leafy Greens', 'Carrot Juice', 'Turmeric', 'Ginger Tea'],
    'Hepatitis E': ['Turmeric', 'Garlic', 'Leafy Greens', 'Beetroot', 'Carrot Juice'],
    'Alcoholic hepatitis': ['Turmeric', 'Ginger Tea', 'Leafy Greens', 'Garlic', 'Beetroot'],
    'Tuberculosis': ['Turmeric', 'Garlic', 'Leafy Greens', 'Ginger Tea', 'Cayenne Pepper'],
    'Common Cold': ['Ginger Tea', 'Chicken Soup', 'Honey', 'Citrus Fruits', 'Garlic'],
    'Pneumonia': ['Garlic', 'Ginger Tea', 'Honey', 'Turmeric', 'Oregano Tea'],
    'Dimorphic hemmorhoids(piles)': ['Fiber-Rich Foods', 'Fluids', 'Green Leafy Vegetables', 'Whole Grains', 'Prunes'],
    'Heart attack': ['Oats', 'Olive Oil', 'Garlic', 'Leafy Greens', 'Berries'],
    'Varicose veins': ['Flaxseeds', 'Ginger', 'Cayenne Pepper', 'Leafy Greens', 'Berries'],
    'Hypothyroidism': ['Seaweed', 'Brazil Nuts', 'Turmeric', 'Ginger', 'Coconut Oil'],
    'Hyperthyroidism': ['Cruciferous Vegetables', 'Yogurt', 'Berries', 'Whole Grains', 'Nuts'],
    'Hypoglycemia': ['Sweet Potatoes', 'Oats', 'Nuts', 'Berries', 'Cinnamon'],
    'Osteoarthritis': ['Turmeric', 'Ginger Tea', 'Leafy Greens', 'Fatty Fish', 'Berries'],
    'Arthritis': ['Ginger', 'Turmeric', 'Berries', 'Pineapple', 'Walnuts'],
    'Vertigo (Paroxysmal Positional Vertigo)': ['Ginger Tea', 'Ginkgo Biloba', 'Almonds', 'Spinach', 'Salmon'],
    'Acne': ['Green Tea', 'Turmeric', 'Fruits', 'Vegetables', 'Nuts'],
    'Urinary tract infection': ['Cranberry Juice', 'Watermelon', 'Yogurt', 'Celery', 'Cucumber'],
    'Psoriasis': ['Turmeric', 'Olive Oil', 'Leafy Greens', 'Fish', 'Nuts'],
    'Impetigo': ['Turmeric', 'Honey', 'Neem Leaves', 'Coconut Oil', 'Aloe Vera Gel'],
    'Anemia': ['Spinach', 'Beetroot', 'Pomegranate', 'Apple', 'Dates'],
    'Ovarian cancer': ['Green Leafy Vegetables', 'Cruciferous Vegetables', 'Berries', 'Whole Grains', 'Flaxseeds'],
    'Endometriosis': ['Turmeric', 'Fenugreek', 'Ginger', 'Green leafy vegetables', 'Fruits'],
    'Polycystic Ovary Syndrome (PCOS)': ['Cinnamon', 'Turmeric', 'Fenugreek', 'Spearmint tea', 'Whole grains'],
    'Ovarian Cysts': ['Turmeric', 'Flaxseeds', 'Green leafy vegetables', 'Fruits', 'Whole grains'],
    'Uterine Fibroids': ['Turmeric', 'Ginger', 'Flaxseeds', 'Green leafy vegetables', 'Whole grains'],
    'Pelvic Inflammatory Disease (PID)': ['Turmeric', 'Ginger', 'Garlic', 'Green leafy vegetables', 'Whole grains'],
    'Breast Cancer': ['Turmeric', 'Garlic', 'Green tea', 'Cruciferous vegetables', 'Fruits'],
    'Ovarian Cancer': ['Turmeric', 'Ginger', 'Garlic', 'Green leafy vegetables', 'Whole grains'],
    'Menopause': ['Soy products', 'Flaxseeds', 'Green leafy vegetables', 'Nuts', 'Whole grains'],
    'Vaginal Yeast Infection': ['Curd (Yogurt)', 'Garlic', 'Fenugreek', 'Turmeric', 'Green leafy vegetables'],
    "Prostate Cancer": ["Cruciferous vegetables (broccoli, cauliflower)", "Tomatoes", "Berries", "Nuts and seeds", "Healthy oils (olive oil)", "Green tea"],
    "Testicular Cancer": ["Fruits and vegetables", "Whole grains (brown rice, oats)", "Lean proteins (poultry, fish, legumes)", "Healthy fats (olive oil, nuts)", "Limit processed foods and sugary snacks"],
    "Benign Prostatic Hyperplasia (BPH)": ["Zinc-rich foods (pumpkin seeds, legumes)", "Antioxidant-rich foods (berries, spinach)", "Healthy fats (avocado, olive oil)", "Lean proteins (chicken, turkey)", "Limit caffeine and alcohol"],
    "Erectile Dysfunction": ["Nitrates-rich foods (leafy greens, beets)", "Omega-3 fatty acids (fish, flaxseeds)", "Whole grains (quinoa, whole wheat)", "Lean proteins (poultry, legumes)", "Limit processed foods and high-fat meals"],
    "Hypothyroidism": ["Iodine-rich foods (iodized salt, seafood)", "Selenium sources (Brazil nuts, whole grains)", "Lean proteins (chicken, fish, tofu)", "Fiber-rich foods (fruits, vegetables, whole grains)", "Limit goitrogenic foods (cabbage, broccoli)"],
    "Testicular Torsion": ["Balanced diet with fruits, vegetables, whole grains, lean proteins", "Vitamin C and E-rich foods (citrus fruits, nuts)", "Healthy fats (avocado, olive oil)", "Limit processed foods and sugary snacks"],
    "Penile Fracture": ["Vitamin C and collagen-rich foods (citrus fruits, bell peppers)", "Lean proteins (poultry, fish)", "Anti-inflammatory foods (turmeric, ginger)", "Whole grains and fiber-rich foods", "Limit processed foods and high-fat meals"],
    "Cryptorchidism": ["Balanced diet with fruits, vegetables, whole grains, lean proteins", "Sources of zinc, selenium, vitamin E", "Healthy fats (avocado, nuts)", "Limit processed foods and high-fat meals"]
    
    
}





