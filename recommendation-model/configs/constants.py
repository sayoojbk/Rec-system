# ------------------------------------------------------------------------------------------------------
# Licensed under MIT License
# Written by sayooj_bk
# -------------------------------------------------------------------------------------------------------

# Default column names

# This is the user meta-data which user provides.
DEFAULT_USER_COL = 'userId'
DEFAULT_USER_BUDGET = "userBudget"
DEFAULT_PURCHASED_COUNT = "purchase_count"
DEFAULT_NATIONALITY = "nationality"


# The product data 
DEFAULT_RETAIL_PRICE = "itemPrice"
DEFAULT_ITEM_COL = "product_id"                           # This is what item user prefers from the list of items available.
DEFAULT_CATEGORY = "itemCategory"



#   The merchant data scheme  
MERCHANT_ID = "merchantId"
MERCHANT_NAME = "merchantName"
LOCATION = "location"



#   Merchant product data
#   - MERCHANT_ID , DEFAULT_ITEM_COL,  DEFAULT_RETAIL_PRICE , DISCOUNTED_PRICE
MERCHANT_PRODUCT_PRICE = "merchantProductPrice"
DISCOUNTED_PRICE = "discountedPrice"

#   Merchant category data
MERCHANT_CATEGORY_ID = "merchantCategoryId"
PRICE_LEVEL = "priceLevel"   # This is the price level of the merchant like is this a expensive place or cheap place like that.



# Filtering variables
DEFAULT_K = 10
DEFAULT_THRESHOLD = 10

# Other
SEED = 42



