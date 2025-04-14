# fashion-styling


Hi Dr. Gerber & TAs, and welcome to our final project!
There are two models that we built: A CNN that aims to predict product type and graphical appearance, and an item-item CF algorithm that aims to provide clothing recommendations based on reviews. 

For our CNN model:
* The articles csv is the data containing articles of clothing with characteristics
* The dataset of files we used was too large to insert, even when zipped, so the file was uploaded to Google Drive. The link is here: https://drive.google.com/drive/folders/1SlcwlD5KQafzmMhu_4Bf1DB_2DSz3rWX?usp=sharing

For our CF model: 
* The Fashion_Retail_Sales.csv is the file we used for the CF model that contains the customers, items, and ratings.
* The 'CollabFilter.py' is the file that contains all the code for only the CF model but is also used for our dashboard.

For our Dashboard: 
* In order to get the dashboard working, you will need the following in the same directory: CollabFilter.py, dashboard.py, the 'static' folder, and the .csv file. However, you only need to run 'dashboard.py' to use the dashboard. When you run dashboard.py, the terminal will display outputs similar to those you would see from running CollabFilter.py, such as the MSE, RMSE, and MAE values. Shortly after, the code will output a message like "Launching server at http://localhost:57134" and automatically open your browser to the dashboard’s landing page. From there, you can click on the 'Item Recommendation Examples' tab to view our interactive CF results. As a heads up, if you terminate the code, the page will stay open, but it will no longer respond correctly. 
