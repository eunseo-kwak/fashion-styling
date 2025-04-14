# Import needed libraries
from CollabFilter import CollabFilter
import panel as pn
import pandas as pd

# Load resources for Panel widgets and plots to work
pn.extension()

# Define landing page content
landing_page_card = pn.Column(
    pn.panel("""
# DS4420 Final Project - Fashion Styling
## Eunseo Kwak and Eleanor Washburn

### Motivation
As two students with retail experience and a high-level understanding of the fashion industry, one of the biggest issues facing customers is choosing styling options with a high
accuracy rate. Currently, personal shopping is majorly done in-person, and many factors can impact the success of such styling appointments, such as the stylist, time
committed, and financial investment. While online stores offer algorithms to help customers find options that are right for their personal style and other constraints,
these algorithms often do not take in previous purchase history, and focus on selling their products versus focusing on the overall satisfaction of the customer. Customers have
to ask for recommendations by going in store, taking time and effort, and can get unlucky with the salesperson that assists them. However, the market for personalized styling
is changing, with a major emphasis on AI-based recommendations that let customers discover their next look without having to leave the house. With retailers majorly shifting to
emphasizing online shopping over brick-and-mortar, styling services are making the same pivot. Thus, our problem statement we aim to address in our project is that customers
who are looking at the plethora of clothing options available online are overwhelmed, and unable to combine purchase history from multiple stores along with their personal
preferences from the comfort of their home.

### Methodology and Data Collection
The data used for the item-item collaborative filtering came from [Kaggle](https://www.kaggle.com/datasets/fekihmea/fashion-retail-sales?select=Fashion_Retail_Sales.csv) that contains a comprehensive collection of data representing sales transactions from a clothing store. 
By leveraging detailed customer interaction data, such as item purchases, reviews, and ratings, the collaborative filtering system uncovers patterns in user behavior, providing 
relevant suggestions based on what other customers like. In particular, our item-item collaborative filtering method identifies relationships between items and uses these 
connections to recommend products that share similarities, ensuring that each recommendation feels personal and intuitive. Complementing this is our CNN model, which enhances 
recommendations by analyzing the visual features of fashion items. The data for our CNN model was also sourced from [Kaggle](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview), specifically from a large dataset published by H&M, 
a Sweden-based fast-fashion retailer with over 4,300 stores worldwide and $22.1 billion in revenue in 2024 alone (Forbes, “H&M Company Overview). H&M produces up to 16 collections 
annually, introducing over 4,400 products in 2022 (Nielsen, “Close-up on the Ultra-Fast Fashion Market). With this high product turnover, providing timely and personalized 
recommendations is critical. To support this, we trained a CNN to extract visual features from product images, such as patterns, colors, and textures, and categorize each item 
by its graphical appearance (e.g., striped, floral, solid). These features help customers filter products by visual preferences and discover items that match their unique style. 
Together, our collaborative filtering system and CNN-based visual model create a fashion recommendation experience. Users can receive suggestions based not only on their ratings 
and preferences but also on the stylistic traits they gravitate toward, offering a more personalized online shopping journey.
"""),
pn.pane.HTML(
f'<img src="https://yatter.in/wp-content/uploads/2024/04/GettyImages-1345105965-2048x1152.webp" width="1100">'),
pn.panel(""" ### References 
Gordenkoff. Beautiful Female Customer Using 3D Augmented Reality Digital Interface in Modern Shopping Center. Shopper is Choosing Fashionable Bags, Stylish Garments in Clothing Store. Futuristic VFX UI Concept. stock photo. October 13, 2021. iStock.
H&M - Hennes & Mauritz | Company Overview & News.” Forbes. Accessed April 13, 2025. https://www.forbes.com/companies/hm-hennes-mauritz/
Shein, Zara, H&M: Close-up on the Ultra-Fast Fashion Market.” Nielsen IQ, June 22, 2023. https://nielseniq.com/global/en/insights/analysis/2023/shein-zara-hm-close-up-on-the-ultra-fast-fashion-market/
"""))

# Load in the fashion retail sales dataset
fashion_df = pd.read_csv("Fashion_Retail_Sales.csv")
# Create an instance of the CollabFilter Class
cf = CollabFilter()
# Prepare and clean the dataset for collaborative filtering
fashion_df = cf.prepare_data(fashion_df)
# Create the user-item matrix
user_item_matrix = cf.create_user_item_matrix(fashion_df)
# Create the similarity score matrix
similarity_matrix = cf.compute_item_similarity_matrix(user_item_matrix)

# Extract unique customer IDs from the dataset for the dropdown selector
unique_customers = fashion_df['Customer Reference ID'].unique().tolist()

# Create a Panel widget to select a customer ID from a dropdown
customer_selector = pn.widgets.Select(name='Choose Customer ID', options=unique_customers)

# Map item names to their corresponding image filenames
item_image_map = {
    'Hat': 'Hat.png',
    'Belt': 'Belt.png',
    'Wallet': 'Wallet.png',
    'Socks': 'Socks.png',
    'Sandals': 'Sandal.png',
    'Sunglasses': 'Sunglasses.png',
    'Trousers': 'Trousers.png',
    'Sweater': 'Sweater.png',
    'Romper': 'Romper.png',
    'Tunic': 'Tunic.png',
    'Onesie': 'Onesie.png',
    'Blouse': 'Blouse.png',
    'Loafers': 'Loafers.png',
    'Tank Top': 'Tank.png',
    'Skirt': 'Skirt.png',
    'Tie': 'Tie.png',
    'Jumpsuit': 'Jumpsuit.png',
    'Coat': 'Coat.png',
    'Blazer': 'Blazer.png',
    'Vest': 'Vest.png',
    'Handbag': 'Handbag.png',
    'Swimsuit': 'Swimsuit.png',
    'Leggings': 'Leggings.png',
    'T-shirt': 'TShirt.png',
    'Gloves': 'Gloves.png',
    'Trench Coat': 'Trench Coat.png',
    'Slippers': 'Slippers.png',
    'Sneakers': 'Sneakers.png',
    'Kimono': 'Kimono.png',
    'Cardigan': 'Cardigan.png',
    'Shorts': 'Shorts.png',
    'Pajamas': 'Pjs.png',
    'Camisole': 'Camisole.png',
    'Pants': 'Pants.png',
    'Overalls': 'Overalls.png',
    'Raincoat': 'Raincoat.png',
    'Scarf': 'Scarf.png',
    'Poncho': 'Poncho.png',
    'Flannel Shirt': 'Flannel Shirt.png',
    'Jacket': 'Jacket.png',
    'Boots': 'Boots.png',
    'Backpack': 'Backpack.png',
    'Bowtie': 'Bowtie.png',
    'Flip-Flops': 'Flip Flops.png',
    'Jeans': 'Jeans.png',
    'Hoodie': 'Hoodie.png',
    'Umbrella': 'Umbrella.png',
    'Sun Hat': 'Sun Hat.png',
    'Dress': 'Dress.png',
    'Polo Shirt': 'Polo Shirt.png'}

# Function to update recommendations whenever a customer is selected
def update_recommendations(customer_id):
    # Predict ratings for the selected user using k nearest items
    ratings = cf.predict_ratings_for_user(user_item_matrix, similarity_matrix, customer_id, k=7)
    # Get the user’s original ratings
    original_ratings = user_item_matrix.loc[customer_id]
    # Get the top 3 recommended items for this customer
    recommended_items = cf.get_top_k_recommendations(ratings, original_ratings, k=3)
    # Start a list of components to build a Panel column
    recommendation_components = [f"### Recommendations for Customer {customer_id}"]

    # Loop through each recommended item to display name, score, and image
    for item, rating in recommended_items.items():
        # Get image filename if exists
        image_path = item_image_map.get(item)
        if image_path:
            # Display image and rating in a horizontal row
            row = pn.Row(
                # Display the recommended item and rating
                pn.pane.Markdown(f"**{item}** — Rating: {rating:.2f}", width=200),
                # Display the corresponding image next to the item and rating
                pn.pane.PNG(f"static/{image_path}", width=150, height=150))
            # Add the row with item info and image to the list of recommendations
            recommendation_components.append(row)
        else:
            # If no image is available, just show the name and rating
            recommendation_components.append(pn.pane.Markdown(f"- {item}: {rating:.2f} (no image)"))

    # Return the column of recommendations
    return pn.Column(*recommendation_components)

# Bind the update function to the customer selector
recommendation_panel = pn.bind(update_recommendations, customer_id=customer_selector)

# Create the recommendations tab with title, selector, and dynamic recommendation panel
recommendations_tab = pn.Column(
    pn.pane.Markdown("## Personalized Recommendations"),
    customer_selector,
    recommendation_panel)

# Create the full layout using Panel's FastListTemplate
layout = pn.template.FastListTemplate(
# Define the dashboard title
    title='Fashion Styling',
    # Disable dark/light theme toggle
    theme_toggle=False,
    main=[
        pn.Tabs(
            # Load in the landing page card
            ('About Our Project', landing_page_card),
            # Load in the recommender card
            ('Item Recommendation Examples', recommendations_tab),
            # Have the dashboard open on the landing page
            active=0)],
    # Define the title background color
    header_background='#BD4F6C')

# Display the layout
layout.servable()
layout.show()