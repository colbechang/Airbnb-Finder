"""
DS2500 Project
Finding the best Airbnb in NYC for your price budget
Colbe Chang

Dataset Used: https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model



ABNB_FILE = "AB_NYC_2019.csv"


def multi_lin_reg_pred(df, min_nights, num_reviews, availability):
    """
    

    Parameters
    ----------
    df : dataframe
        Dataframe containing airbnb data.
    min_nights : int
        minimum number of nights a guest can stay.
    num_reviews : int
        number of reviews for a airbnb.
    availability : int
        number of days that airbnb is available per year.

    Returns
    -------
    prediction : float
        price prediction of an airbnb based on minimum number of nights,
        number of reviews, and availability.

    """
    
    # Setting x values equal to the predictor variables and y equal to 
    # the prediction variable
    x = df[["minimum_nights", "number_of_reviews", 
            "availability_365"]]
    y = df["price"]
    
    # initializing the linear regression and using .fit() to train the model
    # and get the coefficient for the regression equation
    multi_lin_reg = linear_model.LinearRegression()
    multi_lin_reg.fit(x.values, y)
    
    # using the model to predict the price based on those 3 variables
    prediction = multi_lin_reg.predict([[min_nights, num_reviews, 
                                         availability]])
    return prediction

def plot_actual_to_pred(df):
    """
    

    Parameters
    ----------
    df : dataframe
        name of dataframe.

    Returns
    -------
    Graph of actual prices compared to the prices predicted from the same
    data.

    """
    
    actual = []
    pred = []
    
    # initializing figure object
    fig = plt.figure()
    
    # looping through the dataframe
    for i in range(len(df)):
        # appending the price for each row to the actual list
        actual.append(df.loc[i, "price"])
        # getting the prediction of the price from the data from that row
        prediction = multi_lin_reg_pred(df, df.loc[i, "minimum_nights"],
                                        df.loc[i, "number_of_reviews"],
                                        df.loc[i, "availability_365"])
        # appending the prediction to the prediction list
        # (predicition returns a list, so we only want the number from it)
        pred.append(round(prediction[0]))
    
    # Plotting the graph
    ax1 = fig.add_subplot(111)
    ax1.scatter(range(len(actual)), actual, c = "green", label="Actual")
    ax1.scatter(range(len(pred)), pred, c = "red", label="Prediction")
    plt.xlabel("Test Cases")
    plt.ylabel("Price")
    plt.title("Predicted Price vs Actual Price")
    plt.legend()
    plt.show()
    

def price_ranges(price):
    """
    

    Parameters
    ----------
    price : int
        price of airbnb.

    Returns
    -------
    str
        price range that the price falls in.

    """
    
    
    # Creating price groups for the dataset - to be used with .apply()
    if price > 0 and price < 100:
        return "0-99"
    elif price >= 100 and price < 250:
        return "100-249"
    elif price >= 250 and price < 500:
        return "250-499"
    elif price >= 500 and price < 1000:
        return "500-999"
    elif price >= 1000 and price < 2000:
        return "1000-1999"
    else:
        return "Very Expensive"

def borough_dict(df):
    """
    

    Parameters
    ----------
    df : dataframe
        name of dataframe.

    Returns
    -------
    boroughs_prices : dict
        nested dict of boroughs as main keys and within them, the total prices
        of each neighborhood as well as the amount of times they appear
        in the dataset.

    """
    boroughs_prices = {}
    
    # looping through dataframe
    for i in range(len(df)):
        # getting the borough and neighborhood name of the current row
        borough = df.loc[i, "neighbourhood_group"]
        neighborhood = df.loc[i, "neighbourhood"]
        # if the current borough is not in the dict, add it with an empty
        # dict as value
        if borough not in boroughs_prices.keys():
            boroughs_prices[borough] = {}
        # if the current neighborhood is not in the dict of the borough,
        # add it and its price as well as a count for the amount
        # of times it shows up
        elif neighborhood not in boroughs_prices[borough].keys():
            boroughs_prices[borough][neighborhood] = {"price": df.loc[i, "price"]}
            boroughs_prices[borough][neighborhood]["count"] = 1
        # otherwise, add the price to that neighborhood and add 1 to the count
        else:
            boroughs_prices[borough][neighborhood]["price"] += df.loc[i, "price"]
            boroughs_prices[borough][neighborhood]["count"] += 1
        
    return boroughs_prices

def borough_price_means(borough_prices):
    """
    

    Parameters
    ----------
    borough_prices : dict
        nested dict of boroughs as main keys and within them, the total prices
        of each neighborhood as well as the amount of times they appear
        in the dataset.

    Returns
    -------
    borough_means : dict
        nested dict of boroughs as main keys and each of its neighborhoods
        along with the average price of an airbnb in that neighborhood within
        it.

    """
    
    # looping through each borough and neighborhood in the dict
    for boroughs in borough_prices.keys():
        for neighborhoods in borough_prices[boroughs].keys():
            # calculating the mean by taking the total price accumulated
            # in that neighborhood and dividing it by the count of the 
            # amount of times that neighborhood showed up in the dataset
            mean = borough_prices[boroughs][neighborhoods]["price"] / \
                borough_prices[boroughs][neighborhoods]["count"]
            mean = round(mean, 2)
            # removing previous data of price/count and replacing it with mean
            borough_prices[boroughs][neighborhoods] = mean
            
    # setting variable to a different name for clarity
    borough_means = borough_prices
    return borough_means

def plot_neighborhood_avg_prices(borough_means, borough):
    """
    

    Parameters
    ----------
    borough_means : dict
        nested dict of boroughs as main keys and each of its neighborhoods
        along with the average price of an airbnb in that neighborhood within
        it.
    borough : str
        name of a borough.

    Returns
    -------
    graph of the average prices of airbnbs for each neighborhood in that borough.

    """
    neighborhoods = []
    avg_prices = []
    
    # looping through the neighborhoods in the selected borough and
    # appending the name of the neighborhood and its average price to two 
    # separate lists
    for keys in borough_means[borough]:
        neighborhoods.append(keys)
        avg_prices.append(borough_means[borough][keys])
    
    # graphing
    plt.show()
    plt.bar(neighborhoods, avg_prices, align='edge')
    plt.xticks(rotation=90)
    plt.tick_params(axis='x', which='major', labelsize=7)
    plt.xlabel("Neighborhoods")
    plt.ylabel("Average Price")
    plt.title("Average Airbnb Prices for Neighborhoods in {}".format(borough))
   
    
    
def main():
    
 
    df = pd.read_csv(ABNB_FILE)
    
    # adding a column for price ranges in the dataframe
    df["Price Ranges ($)"] = df.apply(lambda row: price_ranges(row["price"]), 
                                  axis=1)
    
    # Plotting a map of all the airbnbs in ny, colorcoded with its 
    # corresponding price range
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x='longitude', y='latitude', hue='Price Ranges ($)', 
                    hue_order=["0-99", "100-249", "250-499",
                               "500-999", "1000-1999", "Very Expensive"], 
                    data=df)
    plt.title("Prices of Airbnbs in NYC")
    
    
    boroughs = borough_dict(df)
    borough_means = borough_price_means(boroughs)
    
    # asking user to name a borough to plot average prices of each neighborhood
    # in that borough
    borough = input("Name a borough in NYC: ")
    if borough.capitalize() != "Staten island":
        borough = borough.capitalize()
    else:
        words = borough.split()
        borough = words[0].capitalize() + " " + words[1].capitalize()
    plot_neighborhood_avg_prices(borough_means, borough)
    
    """
    This is used to plot the multi linear regression actual vs prediction 
    graph, but it takes a while to load. Uncomment to see it
    """
    #plot_actual_to_pred(df)
    
if __name__ == "__main__":
    main()