import flet as ft
import json
import os
import random
import bcrypt
from datetime import datetime
import re
import numpy as np
import yfinance as yf
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from flet.matplotlib_chart import MatplotlibChart
from stable_baselines3 import PPO
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
import zipfile
import torch
import schedule
import time
import threading


matplotlib.use("svg")


class State:
    toggle = True

s = State()

# File to store user data
USER_DATA_FILE = "user_data.json"

# Load users from the JSON file
def load_users():
    if os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "r") as file:
            return json.load(file)
    return {}



# Save users to the JSON file
def save_users():
    with open(USER_DATA_FILE, "w") as file:
        json.dump(users_db, file)

# Load users at startup
users_db = load_users()

current_user = None  




def get_current_data():
        stock_data = yf.Ticker('^GSPC')
        history = stock_data.history(period="50d")  # Fetch 30 days of data for calculations
        
        # Current day data
        stock_price = history["Close"].iloc[-1] 
        volume = history["Volume"].iloc[-1]  
        
        # Calculate technical indicators and extract the last value
        ma_50 = history["Close"].rolling(50).mean().iloc[-1] 
        ma_20 = history["Close"].rolling(20).mean().iloc[-1]   
        rsi_30 = (100 - (100 / (1 + history["Close"].pct_change().rolling(30).apply(lambda x: (x[x > 0].mean() / abs(x[x <= 0].mean())) if x[x <= 0].mean() != 0 else np.inf)))).iloc[-1]  
        
        return stock_price, volume, ma_20, ma_50, rsi_30


# Function to update user portfolio based on action
def update_user_portfolio(action, stock_price, current_user):
    action = action[0]
    user_info = users_db.get(current_user, {})
    portfolio_size = user_info.get("portfolio_size", 0.0)
    shares_owned = user_info.get("shares_owned", 0)
    
    percent = 0.2 * action

    if action < 0.5:  # Buy
        amount_to_spend = portfolio_size * percent
        number_of_shares = amount_to_spend / stock_price
        users_db[current_user]["shares_owned"] += number_of_shares
        users_db[current_user]["portfolio_size"] -= number_of_shares * stock_price

    elif action > 0.5:  # Sell
        number_of_shares_to_sell = shares_owned * percent
        users_db[current_user]["shares_owned"] -= number_of_shares_to_sell
        users_db[current_user]["portfolio_size"] += number_of_shares_to_sell * stock_price

    save_users()


def predict_performance():
    zip_path = "agent.zip"

    
    try:
        trained_model = PPO.load(zip_path)
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return  # Stop execution if model fails to load

    if current_user is None:
        print("⚠️ No user logged in!")
        return

    # Get stock data
    stock_price, volume, ma_20, ma_50, rsi_30 = get_current_data()

    # Get user portfolio
    user_info = users_db.get(current_user, {})
    portfolio_size = user_info.get("portfolio_size", 0.0)
    shares_owned = user_info.get("shares_owned", 0)

    # Calculate total portfolio value
    total_value = portfolio_size + (shares_owned * stock_price)

    # Construct observation for AI model
    observation = [total_value, stock_price, volume, ma_20, ma_50, rsi_30]

    # Make prediction (buy, sell, hold)
    action, _ = trained_model.predict(observation, deterministic=True)
    print(f"📈 Predicted Action: {action}")

    # Update user portfolio
    update_user_portfolio(action, stock_price, current_user)


    print("performance predicted :)")



# Function to start scheduler in a separate thread
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60)

# Schedule predict_performance to run daily at 8 AM
schedule.every().day.at("08:00").do(lambda: predict_performance()) 

# Start the scheduler in a new thread
scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
scheduler_thread.start()









def main(page: ft.Page):
    page.scroll = True
    page.bgcolor = "#ffffff"

    global current_user  # Allow modifying the global variable

    gradient = ft.LinearGradient(
        begin=ft.alignment.top_center,
        end=ft.alignment.bottom_center,
        colors=[ft.colors.WHITE, ft.colors.BLUE],
    )

    def show_home(page):
        page.controls.clear()

        graph = ft.Image(
            src="graph1.png", 
            fit=ft.ImageFit.COVER,            
            opacity=0.6,
            height=page.height,
            width=page.width,
        )

        background = ft.Container(
            content=graph,
            expand=True,
        )

        if current_user:
            user_info = users_db.get(current_user, {})
            name = user_info.get("name", "Unknown User")
            portfolio_size = user_info.get("portfolio_size", 0.0)

            stock_price, _, _, _, _ = get_current_data()
            shares_owned = user_info.get("shares_owned", 0.0)
            shares_value = shares_owned * stock_price
            total_value = portfolio_size + shares_value

            home_logged_in = ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text(f"Welcome, {name}!", size=40, weight="bold", color="#1e3a8a"),
                        ft.Text(f"Your Portfolio Size: ${total_value:,.2f}", size=20, color="#34495e"),
                    ],
                    alignment=0.4,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=20,
                ),
                height=page.height,
                width=page.width,
                bgcolor="#ffffff",
                gradient=gradient,
                padding=ft.padding.all(20),
            )

            def on_hover(e):
                if e.data == "true":  # Mouse hover
                    e.control.bgcolor = "#3b82f6"  # Lighter blue
                    e.control.scale = 1.1  # Scale up slightly
                else:  # Mouse leaves
                    e.control.bgcolor = "#1e40af"  # Default dark blue
                    e.control.scale = 1.0  # Reset scale
                e.control.update()

            buttons = ft.Row(
                controls=[
                    ft.Container(
                        content=ft.Column(
                            controls=[
                                ft.Icon(ft.icons.SHOW_CHART, size=50, color="white"),
                                ft.Text("View Performance", size=20, weight="bold", color="white", text_align=ft.TextAlign.CENTER),
                                ft.Text("Check your investment growth", size=14, color="white"),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        on_click=lambda _: show_performance(page),
                        height=200,
                        width=150,
                        bgcolor="#1e40af",
                        border_radius=30,
                        padding=ft.padding.all(15),
                        on_hover=on_hover,
                        animate_size=ft.animation.Animation(200, ft.AnimationCurve.EASE_OUT), 
                    ),
                    ft.Container(
                        content=ft.Column(
                            controls=[
                                ft.Icon(ft.icons.ACCOUNT_BALANCE_WALLET, size=50, color="white"),
                                ft.Text("Deposit Money", size=20, weight="bold", color="white", text_align=ft.TextAlign.CENTER),
                                ft.Text("Add funds to your account", size=14, color="white"),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        on_click=lambda _: show_deposit(page),
                        height=200,
                        width=150,
                        bgcolor="#1e40af",
                        border_radius=30,
                        padding=ft.padding.all(15),
                        on_hover=on_hover,
                        animate_size=ft.animation.Animation(200, ft.AnimationCurve.EASE_OUT), 
                    ),
                    ft.Container(
                        content=ft.Column(
                            controls=[
                                ft.Icon(ft.icons.ATM, size=50, color="white"),
                                ft.Text("Withdraw Money", size=20, weight="bold", color="white", text_align=ft.TextAlign.CENTER),
                                ft.Text("Retrieve funds from account", size=14, color="white"),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        on_click=lambda _: show_withdraw(page),
                        height=200,
                        width=150,
                        bgcolor="#1e40af",
                        border_radius=30,
                        padding=ft.padding.all(15),
                        on_hover=on_hover,
                        animate_size=ft.animation.Animation(200, ft.AnimationCurve.EASE_OUT), 
                    ),
                    ft.Container(
                        content=ft.Column(
                            controls=[
                                ft.Icon(ft.icons.PIE_CHART, size=50, color="white"),
                                ft.Text("Portfolio Breakdown", size=20, weight="bold", color="white", text_align=ft.TextAlign.CENTER),
                                ft.Text("View detailed asset info", size=14, color="white"),
                            ],
                            alignment=ft.MainAxisAlignment.CENTER,
                            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                        ),
                        on_click=lambda _: show_portfolio_breakdown(page),
                        height=200,
                        width=150,
                        bgcolor="#1e40af",
                        border_radius=30,
                        padding=ft.padding.all(15),
                        on_hover=on_hover,
                        animate_size=ft.animation.Animation(200, ft.AnimationCurve.EASE_OUT), 
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20,
            )

            content_stack = ft.Stack(
                controls=[home_logged_in, background, buttons],
                expand=True,
                alignment=ft.alignment.center,
            )
            page.add(navbar, content_stack)

        else:
            home_section = ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("Smart Investments with AI Investing", size=56, weight="bold", color="#1e3a8a"),
                        ft.Text("Let AI make informed decisions and grow your wealth. Your financial future starts here.", size=20, color="#34495e"),
                        ft.ElevatedButton(
                            "Get Started",
                            style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                            on_click=lambda _: show_register(page),
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                height=page.height,
                width=page.width,
                bgcolor="#ffffff",
                gradient=gradient,
                padding=ft.padding.all(20),
            )

            content_stack = ft.Stack(
                controls=[background, home_section],
                expand=True,
            )
            page.add(navbar, content_stack)

        page.update()



    def show_sp500(page):
        # Fetch S&P 500 data for the last year
        page.controls.clear()
        page.bgcolor = ft.colors.WHITE  # Set page background to white
        
        # Define the ticker for the S&P 500 index
        ticker = "^GSPC"  # S&P 500 index symbol

        # Download the last year of daily data
        data = yf.download(ticker, period="1y", interval="1d")

        # Check if data is empty
        if data.empty:
            raise ValueError("Failed to fetch S&P 500 data.")

        # Extract the 'Close' column and ensure it's a Series
        close_prices = data["Close"].squeeze()
        close_prices = close_prices.dropna()  # Drop any NaN values
        prices_list = close_prices.tolist()

        #get the min and max 
        min_y, max_y = close_prices.min(), close_prices.max()


        text = ft.Column(
            controls=[
                ft.Divider(),
                ft.Text(
                    "This graph shows the daily closing prices of the S&P 500 index over the past year. "
                    "The S&P 500 is a major benchmark for the overall US stock market performance.",
                    size=18,
                    color=ft.colors.BLUE_GREY,
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.Text(
                    f"📊 Current Price: ${prices_list[-1]:.2f}  |  📉 1-Year Low: ${min_y:.2f}  |  📈 1-Year High: ${max_y:.2f}",
                    size=16,
                    color=ft.colors.BLACK,
                    weight="bold",
                    text_align=ft.TextAlign.CENTER,
                ),
                ft.ElevatedButton(
                    "Learn More",
                    style=ft.ButtonStyle(bgcolor=ft.colors.BLUE, color=ft.colors.WHITE),
                    on_click=lambda _: show_moreAbout(page),
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=10,
        )

        fig, axs = plt.subplots()
        axs.plot(close_prices.index, close_prices.values, label="S&P 500 Close Price", color="blue")

        # Format x-axis to show every other month
        axs.xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # Every 2 months
        axs.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  # Format as 'Jan 2024', 'Mar 2024', etc.

        axs.set_xlabel("Time")
        axs.set_ylabel("Price")
        axs.grid(True)
        axs.legend()
        # plt.xticks(rotation=45)  # Rotate labels for better readability


        # Add components to the page
        page.title = "S&P 500 - Last Year's Data"
        page.add(
            navbar,
            ft.Row(
                [
                    ft.Text("S&P 500 Performance - Last Year", size=24, weight="bold", color="#1e3a8a"),
                ],
                alignment=ft.MainAxisAlignment.CENTER  
            ),
            ft.Divider(),
            # ft.Container(MatplotlibChart(fig, expand=True), width=800, height=600, alignment=ft.alignment.center),
            ft.Row(
                [
                    ft.Container(
                        MatplotlibChart(fig, expand=True),
                        width=800,
                        height=600
                    )
                ],
                alignment=ft.MainAxisAlignment.CENTER  # Centers horizontally
            ),
            text
        )
        page.update()
    



    
        


    # Function to show performance and make predictions
    def show_performance(page):
        zip_path = "/Users/henryforrest/Documents/Computer Science Work/Dissertation/flet/Dissertation/agent.zip"

        try:
            trained_model = PPO.load(zip_path)
            message = "Model loaded successfully!"
            color = ft.colors.GREEN
        except Exception as e:
            message = f"Error loading model: {str(e)}"
            color = ft.colors.RED
            trained_model = None

        def predict_performance(e):
            if trained_model is None:
                result_text.value = "Model not loaded. Cannot make predictions."
                result_text.color = ft.colors.RED
            else:
                user_info = users_db.get(current_user, {})
                portfolio_size = user_info.get("portfolio_size", 0.0)
                shares_owned = user_info.get("shares_owned", 0)
                
                # Get current data with technical indicators
                stock_price, volume, ma_20, ma_50, rsi_30 = get_current_data()

                # Calculate total portfolio value
                total_value = portfolio_size + (shares_owned * stock_price)
                
                # Construct the observation (state space)
                observation = [
                    total_value,  # User's current balance 
                    stock_price,  # Current stock price 
                    volume,  # Number of shares traded yesterday 
                    ma_20,  # 20-day moving average 
                    ma_50,  # 50-day moving average 
                    rsi_30,  # 30-day RSI 
                ]
                
                # Debug: Print observation values
                print(f"Observation: {observation}")
                
                # Make prediction (buy, sell, hold)
                action, _ = trained_model.predict(observation, deterministic=True)
                print(f"Predicted Action: {action}")
                
                # Update user portfolio based on action
                update_user_portfolio(action, stock_price, current_user)
                
                result_text.value = f"Predicted Action: {action}. Portfolio Updated!"
                result_text.color = ft.colors.BLUE

            page.update()

        result_text = ft.Text(value="", size=20)

        performance_page = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Performance Analysis", size=40, weight="bold", color="#1e3a8a"),
                    ft.Text(message, color=color, size=20),
                    ft.ElevatedButton("Predict Performance", on_click=predict_performance),
                    result_text,
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=page.height,
            width=page.width,
            bgcolor="#ffffff",
            padding=ft.padding.all(20),
        )

        page.controls.clear()
        page.add(navbar, performance_page)
        page.update()



    def show_withdraw(page):
        def handle_withdraw(e):
            amount = withdraw_field.value.strip()  # Remove spaces

            if not amount:  # Check if input is empty
                page.open(ft.SnackBar(
                    ft.Text("Amount cannot be empty.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                ))
            else:
                try:
                    amount = float(amount)
                    
                    if amount <= 0:  # Ensure amount is greater than 0
                        page.open(ft.SnackBar(
                            ft.Text("Amount must be greater than $0.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                        ))
                    elif current_user:
                        user_balance = users_db[current_user]["portfolio_size"]
                        
                        if amount > user_balance:  # Prevent overdraft
                            page.open(ft.SnackBar(
                                ft.Text("Insufficient funds. Cannot withdraw more than your balance.", color=ft.colors.WHITE),
                                bgcolor=ft.colors.RED
                            ))
                        else:
                            users_db[current_user]["portfolio_size"] -= amount
                            save_users()
                            page.open(ft.SnackBar(
                                ft.Text(f"Withdrawal of ${amount:.2f} successful!", color=ft.colors.WHITE),
                                bgcolor=ft.colors.GREEN
                            ))
                            show_home(page)
                    else:
                        page.open(ft.SnackBar(
                            ft.Text("You must be logged in to withdraw money.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                        ))
                except ValueError:  # Handle non-numeric inputs
                    page.open(ft.SnackBar(
                        ft.Text("Please enter a valid numeric amount.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                    ))

            page.snack_bar.open = True
            page.update()

        withdraw_field = ft.TextField(
            label="Amount to Withdraw ($)",
            width=300,
            bgcolor=ft.colors.WHITE,
            keyboard_type=ft.KeyboardType.NUMBER,  # Restricts input to numbers on mobile
        )
        
        users_info = users_db.get(current_user, {})
        cash = users_info.get("portfolio_size", 0.0)

        withdraw_page = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Withdraw Money", size=40, weight="bold", color="#1e3a8a"),
                    ft.Text(f"Cash Available to Withdraw: ${cash:,.2f}"),
                    withdraw_field,
                    ft.ElevatedButton(
                        "Withdraw",
                        style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                        on_click=handle_withdraw,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20,
            ),
            height=page.height,
            width=page.width,
            bgcolor="#ffffff",
            padding=ft.padding.all(20),
            gradient=gradient,
        )
        
        page.controls.clear()
        page.add(navbar, withdraw_page)
        page.update()



    def show_deposit(page):
        def handle_deposit(e):
            amount = deposit_field.value.strip()  # Remove leading/trailing spaces
            
            if not amount:  # Check if input is empty
                page.open(ft.SnackBar(
                    ft.Text("Amount cannot be empty.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                ))
            else:
                try:
                    amount = float(amount)
                    if amount <= 0:  # Ensure amount is greater than 0
                        page.open( ft.SnackBar(
                            ft.Text("Amount must be greater than $0.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                        ))
                    elif current_user:
                        users_db[current_user]["portfolio_size"] += amount
                        save_users()
                        page.open(ft.SnackBar(
                            ft.Text(f"Deposit of ${amount:.2f} successful!", color=ft.colors.WHITE),
                            bgcolor=ft.colors.GREEN
                        ))
                        show_home(page)  
                    else:
                        page.open(ft.SnackBar(
                            ft.Text("You must be logged in to deposit money.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                        ))
                except ValueError:  # Catch invalid numbers (e.g., letters, symbols)
                    page.open(ft.SnackBar(
                        ft.Text("Please enter a valid numeric amount.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                    ))
            
            page.snack_bar.open = True
            page.show_snack_bar()
            page.update()

        deposit_field = ft.TextField(
            label="Amount to Deposit ($)",
            width=300,
            bgcolor=ft.colors.WHITE,
            keyboard_type=ft.KeyboardType.NUMBER,  # Restricts input to numbers on mobile
        )

        deposit_page = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Deposit Money", size=40, weight="bold", color="#1e3a8a"),
                    deposit_field,
                    ft.ElevatedButton(
                        "Deposit",
                        style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                        on_click=handle_deposit,
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=20,
            ),
            height=page.height,
            width=page.width,
            bgcolor="#ffffff",
            padding=ft.padding.all(20),
            gradient=gradient,
        )
        
        page.controls.clear()
        page.add(navbar, deposit_page)
        page.update()




    def handle_logout(e):
        global current_user
        current_user = None
        page.open(ft.SnackBar(ft.Text("Logout successful!"), bgcolor=ft.colors.GREEN))
        page.snack_bar.open = True
        page.update()
        show_home(page)

    def show_account(page):
        
        if current_user:
            user_info = users_db.get(current_user, {})
            name = user_info.get("name", "Unknown User")
            portfolio_size = user_info.get("portfolio_size", 0.0)

            def handle_update(e):
                updated_name = name_field.value.strip()
                if updated_name:
                    users_db[current_user]["name"] = updated_name
                    save_users()
                    page.open(ft.SnackBar(
                        ft.Text("Profile updated successfully!", color=ft.colors.WHITE), bgcolor=ft.colors.GREEN
                    ))
                    show_account(page)
                else:
                    page.open(ft.SnackBar(
                        ft.Text("Name cannot be empty.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                    ))
                page.snack_bar.open = True
                page.update()

            name_field = ft.TextField(label="Name", value=name, width=300, bgcolor=ft.colors.WHITE)

            shares_owned = users_db[current_user]["shares_owned"]
            stock_price, _, _, _, _ = get_current_data()
            shares_value = shares_owned * stock_price
            total_value = portfolio_size + shares_value

            account_page = ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text("Account Information", size=40, weight="bold", color="#1e3a8a"),
                        ft.Divider(),
                        ft.Text(f"Email: {current_user}", size=20, color="#34495e"),
                        ft.Text(f"Portfolio Size: ${total_value:,.2f}", size=20, color="#34495e"),
                        ft.Divider(),
                        ft.Text("Update Profile", size=24, weight="bold", color="#1e3a8a"),
                        name_field,
                        ft.ElevatedButton(
                            "Update Profile",
                            style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                            on_click=handle_update,
                        ),
                        ft.Divider(),
                        ft.ElevatedButton(
                            "Log Out",
                            style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                            on_click=handle_logout,
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    spacing=20,
                ),
                height=page.height,
                width=page.width,
                bgcolor="#ffffff",
                padding=ft.padding.all(20),
                gradient=gradient,
            )
            page.controls.clear()
            page.add(navbar, account_page)
        else:
            page.controls.clear()

            no_account_page = ft.Container(
                content=ft.Column(
                    controls=[
                        ft.Text(
                            "You are not logged in. Please log in to view your account information.",
                            size=20,
                            color=ft.colors.RED,
                            text_align=ft.TextAlign.CENTER,  # Align text centrally within the column
                        ),
                        ft.ElevatedButton(
                            "Login",
                            style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                            on_click=lambda _: show_login(page),
                        ),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,  # Centre controls vertically
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,  # Centre controls horizontally
                    spacing=10,  # Add spacing between the text and button
                ),
                alignment=ft.alignment.center,  # Centre the entire container on the page
                expand=True,  # Ensure the container takes the full page height and width
                padding=ft.padding.all(20),  # Add padding to the container
                bgcolor="#ffffff",  # Set the background colour of the page
                height=page.height,
                width=page.width,
                gradient=gradient,
            )

            page.add(navbar, no_account_page)
            page.update()


    def show_about(page):
        """
        Displays the About page with information about the AI Investing platform.
        Explains how reinforcement learning is used for stock trading.
        """
        about_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("About AI Investing", size=40, weight="bold", color="#1e3a8a"),
                    ft.Divider(),
                    ft.Text(
                        "AI Investing is a cutting-edge platform designed to revolutionise the way you invest in stocks and indices.",
                        size=20, color="#34495e"
                    ),
                    ft.Text(
                        "Using advanced reinforcement learning algorithms, our platform simulates thousands of potential market scenarios to develop strategies that aim to outperform traditional benchmarks like the S&P 500.",
                        size=20, color="#34495e"
                    ),
                    ft.Text(
                        "How it works:", size=24, weight="bold", color="#1e3a8a"
                    ),
                    
                    ft.Text(
                        spans=[
                            ft.TextSpan("1. Data Collection: ", style=ft.TextStyle(weight="bold")),
                            ft.TextSpan("Historical market data and current trends are analysed to identify patterns and opportunities.\n"),
                            ft.TextSpan("2. Learning Process: ", style=ft.TextStyle(weight="bold")),
                            ft.TextSpan("Our reinforcement learning model trains by simulating trading decisions and receiving rewards for profitable actions.\n"),
                            ft.TextSpan("3. Decision-Making: ", style=ft.TextStyle(weight="bold")),
                            ft.TextSpan("The AI autonomously buys and sells shares, aiming to maximise returns while managing risk.\n"),
                            ft.TextSpan("4. Continuous Improvement: ", style=ft.TextStyle(weight="bold")),
                            ft.TextSpan("The model adapts to new market conditions, ensuring its strategies remain effective over time."),
                        ],
                        size=20,
                        color="#34495e",
                    ),

                    ft.Text(
                        "Our mission is to empower users with an accessible and intelligent investment tool, providing peace of mind and confidence in your financial journey.",
                        size=20, color="#34495e", text_align=ft.TextAlign.CENTER
                    ),
                    
                    ft.ElevatedButton(
                        "Learn More",
                        style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                        on_click=lambda _: show_moreAbout(page),
                    ),
                    ft.ElevatedButton(
                        "Glossary",
                        style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                        on_click=lambda _: show_glossary(page),
                    ),
                ],
                alignment=ft.MainAxisAlignment.START,
                spacing=20,
                horizontal_alignment=ft.CrossAxisAlignment.START,
                scroll=ft.ScrollMode.AUTO,  # Enable vertical scrolling
            ),
            expand=True,  # Ensure the container fills the available page space
            padding=ft.padding.all(20),
            bgcolor="#ffffff",
        )

        page.controls.clear()
        page.add(navbar, about_content)
        page.update()

    
    def show_glossary(page):
        """
        Displays informational definitions of words, phrases, and terms that beginners may not know about
        from the investing and reinforcement learning worlds.
        """
        
        glossary_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Glossary", size=40, weight="bold", color="#1e3a8a"),
                    ft.Divider(),

                    # Reinforcement Learning Terms
                    ft.Text("Reinforcement Learning Terms", size=24, weight="bold", color="#1e3a8a"),
                    
                    ft.Text("Agent", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("An entity that interacts with an environment and takes actions to maximise rewards.", 
                            size=18, color="#34495e"),
                    
                    ft.Text("Environment", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("The system or world in which an agent operates, defining states and rewards.", 
                            size=18, color="#34495e"),
                    
                    ft.Text("Reward", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A numerical value given as feedback to an agent to indicate the quality of an action.", 
                            size=18, color="#34495e"),
                    
                    ft.Text("Policy", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A strategy or mapping of states to actions that an agent follows.", 
                            size=18, color="#34495e"),

                    ft.Text("Q-Learning", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A reinforcement learning algorithm that uses a Q-table to learn the best action to take in each state.", 
                            size=18, color="#34495e"),
                    
                    ft.Text("Deep Q-Learning", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("An extension of Q-Learning that uses deep neural networks to approximate Q-values for complex problems.", 
                            size=18, color="#34495e"),

                    ft.Text("Exploration vs. Exploitation", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("The trade-off between trying new actions to gather information (exploration) and choosing actions that are known to yield high rewards (exploitation).", 
                            size=18, color="#34495e"),

                    ft.Divider(),

                    # Investing Terms
                    ft.Text("Investing Terms", size=24, weight="bold", color="#1e3a8a"),
                    
                    ft.Text("ETF (Exchange-Traded Fund)", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A type of investment fund that holds a basket of assets and is traded on the stock exchange like a stock.", 
                            size=18, color="#34495e"),
                    
                    ft.Text("Index", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A way to measure the performance of a group of assets, such as stocks, bonds, or commodities.", 
                            size=18, color="#34495e"),

                    ft.Text("Short Selling", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A trading strategy where an investor borrows a stock, sells it, and hopes to buy it back at a lower price to make a profit.", 
                            size=18, color="#34495e"),

                    ft.Text("Market Capitalisation", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("The total value of a company’s outstanding shares, calculated as: (share price * number of outstanding shares).", 
                            size=18, color="#34495e"),

                    ft.Text("Volatility", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A measure of how much an asset's price fluctuates over time. Higher volatility means higher risk.", 
                            size=18, color="#34495e"),
                    
                    ft.Text("Dividend", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A portion of a company's earnings distributed to shareholders, usually as cash or additional shares.", 
                            size=18, color="#34495e"),
                    
                    ft.Text("Liquidity", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("How quickly an asset can be bought or sold in the market without significantly affecting its price.", 
                            size=18, color="#34495e"),

                    ft.Text("Bear Market", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A prolonged period of declining asset prices, usually defined as a drop of 20% or more.", 
                            size=18, color="#34495e"),

                    ft.Text("Bull Market", size=20, weight="bold", color="#1e3a8a"),
                    ft.Text("A market condition where asset prices are rising or expected to rise.", 
                            size=18, color="#34495e"),
                ],
                alignment=ft.MainAxisAlignment.START,
                spacing=10,
                horizontal_alignment=ft.CrossAxisAlignment.START,
                scroll=ft.ScrollMode.AUTO,  # Enables vertical scrolling
            ),
            expand=True,  # Ensure the container fills the available page space
            padding=ft.padding.all(20),
            bgcolor="#ffffff",
        )

        page.controls.clear()
        page.add(navbar, glossary_content)
        page.update()


    def show_moreAbout(page):
        """
        Displays more detailed information about Reinforcement Learning and the S&P500, ensuring users understand how their money is managed.
        """
        
        def show_rl_content(e):
            content_container.content = rl_content
            page.update()
        
        def show_sp500_content(e):
            content_container.content = sp500_content
            page.update()
        
        about_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("More Info", size=40, weight="bold", color="#1e3a8a"),
                    ft.Divider(),
                    ft.Row(
                        controls=[
                            ft.ElevatedButton(
                                "Reinforcement Learning", 
                                style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                                on_click=show_rl_content
                            ),
                            ft.ElevatedButton(
                                "S&P500", 
                                style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                                on_click=show_sp500_content
                            ),
                        ],
                        alignment=ft.MainAxisAlignment.START,
                    ),
                    ft.Divider(),
                ],
                alignment=ft.MainAxisAlignment.START,
                spacing=20,
                horizontal_alignment=ft.CrossAxisAlignment.START,
                scroll=ft.ScrollMode.AUTO,
            ),
            expand=True,
            padding=ft.padding.all(20),
            bgcolor="#ffffff",
        )
        
        rl_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("What is Reinforcement Learning?", size=28, weight="bold", color="#1e3a8a"),
                    ft.Text(
                        "Reinforcement Learning (RL) is a type of machine learning where an agent learns to make decisions by performing actions in an environment to maximise a reward. "
                        "It is inspired by how humans and animals learn from trial and error. The agent receives feedback in the form of rewards or penalties based on its actions, "
                        "and over time, it learns the best policy to achieve its goal.",
                        size=20,
                        color="#34495e",
                    ),
                    ft.Image(src="RL.png", width=400, height=300, fit=ft.ImageFit.CONTAIN),
                    ft.Text("How Does It Work?", size=24, weight="bold", color="#1e3a8a"),
                    ft.Text(
                        "1. The agent observes the current state of the environment.\n"
                        "2. It takes an action based on its current policy.\n"
                        "3. The environment transitions to a new state, and the agent receives a reward or penalty.\n"
                        "4. The agent updates its policy to improve future decisions.\n"
                        "This process repeats until the agent learns the optimal strategy.",
                        size=20,
                        color="#34495e",
                    ),
                    ft.Text("Why Use RL for Investing?", size=24, weight="bold", color="#1e3a8a"),
                    ft.Text(
                        "Reinforcement Learning is ideal for investing because it can adapt to changing market conditions. "
                        "It learns from historical data and continuously improves its strategies to maximise returns while minimising risks. "
                        "This ensures your money is managed intelligently and efficiently.",
                        size=20,
                        color="#34495e",
                    ),
                ],
                spacing=20,
                scroll=ft.ScrollMode.AUTO,
            ),
        )
        
        sp500_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("What is the S&P500?", size=28, weight="bold", color="#1e3a8a"),
                    ft.Text(
                        "The S&P500, or Standard & Poor's 500, is a stock market index that tracks the performance of 500 of the largest companies listed on U.S. stock exchanges. "
                        "It is widely regarded as one of the best indicators of the overall health of the U.S. stock market and economy, and therefore the world as The New York Stock Exchange (NYSE) is the largest stock exchange in the world and the U.S. economy is the largest in the world.",
                        size=20,
                        color="#34495e",
                    ),
                    ft.Text("Why is the S&P500 Important?", size=24, weight="bold", color="#1e3a8a"),
                    ft.Text(
                        spans=[
                            ft.TextSpan("1. Diversification: ", style=ft.TextStyle(weight="bold")),
                            ft.TextSpan("The S&P500 includes companies from various sectors, reducing the risk of investing in a single industry.\n"),
                            ft.TextSpan("2. Stability: ", style=ft.TextStyle(weight="bold")),
                            ft.TextSpan("The index is composed of large, established companies with a history of strong performance.\n"),
                            ft.TextSpan("3. Growth Potential: ", style=ft.TextStyle(weight="bold")),
                            ft.TextSpan("Over the long term, the S&P500 has consistently provided solid returns for investors.\n"),
                            ft.TextSpan("4. Benchmark: ", style=ft.TextStyle(weight="bold")),
                            ft.TextSpan("It is often used as a benchmark to compare the performance of individual stocks or investment portfolios."),
                        ],
                        size=20,
                        color="#34495e",
                    ),
                    ft.Text("Why Invest in the S&P500?", size=24, weight="bold", color="#1e3a8a"),
                    ft.Text(
                        "Investing in the S&P500 is a smart choice for beginners and experienced investors alike. "
                        "It offers exposure to a broad range of industries and companies, reducing risk while providing the potential for steady growth. "
                        "Additionally, it is a low-cost way to invest in the stock market, as many index funds and ETFs track the S&P500.",
                        size=20,
                        color="#34495e",
                    ),
                ],
                spacing=20,
                scroll=ft.ScrollMode.AUTO,
            ),
        )

        
        content_container = ft.Container(content=rl_content, expand=True)
        page.controls.clear()
        page.add(navbar, about_content, content_container)
        page.update()


    def show_portfolio_breakdown(page):
        page.controls.clear()
        
        # Get user information
        user_info = users_db.get(current_user, {})
        portfolio_size = user_info.get("portfolio_size", 0.0)
        shares_owned = user_info.get("shares_owned", 0)
        
        # Get current stock price
        stock_price, _, _, _, _ = get_current_data()
        shares_value = shares_owned * stock_price
        total_value = portfolio_size + shares_value
        
        # Create portfolio breakdown container
        portfolio_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Portfolio Breakdown", size=40, weight="bold", color="#1e3a8a"),
                    ft.Divider(),
                    ft.Text(f"Total Portfolio Value: ${total_value:,.2f}", size=25, color="#34495e"),
                    ft.Text(f"Available Cash: ${portfolio_size:,.2f}", size=20, color="#2c3e50"),
                    ft.Text(f"Shares Held: {shares_owned}", size=20, color="#2c3e50"),
                    ft.Text(f"Current Stock Price: ${stock_price:,.2f}", size=20, color="#2c3e50"),
                    ft.Text(f"Total Value of Shares: ${shares_value:,.2f}", size=20, color="#2c3e50"),
                    ft.Divider(),
                    ft.ElevatedButton(
                        text="Withdraw Money",
                        on_click=lambda e: show_withdraw(page),
                        bgcolor="#1e3a8a",
                        color="#ffffff"
                    ),
                    ft.ElevatedButton(
                        text="Sell Shares",
                        on_click=lambda e: show_sell_shares(page),
                        bgcolor="#1e3a8a",
                        color="#ffffff"
                    ),
                    ft.ElevatedButton(
                        text="Back to Home",
                        on_click=lambda e: show_home(page),
                        bgcolor="#1e3a8a",
                        color="#ffffff"
                    ),
                ],
                alignment=ft.MainAxisAlignment.START,
                spacing=20,
                horizontal_alignment=ft.CrossAxisAlignment.START,
                scroll=ft.ScrollMode.AUTO,
            ),
            height=page.height,
            width=page.width,
            expand=True,  
            padding=ft.padding.all(20),
            bgcolor="#f4f6f9",
            gradient=gradient,
            
        )

        # Add navbar and portfolio content to the page
        page.add(navbar, portfolio_content)
        page.update()


    def show_sell_shares(page):
        page.controls.clear()
        shares = users_db[current_user]["shares_owned"]
        stock_price, _, _, _, _ = get_current_data()

        def sell_shares():

            amount = deposit_field.value.strip()

            if not amount:  # Check if input is empty
                page.open(ft.SnackBar(
                    ft.Text("Amount cannot be empty.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                ))
            else:
                try:
                    amount = float(amount)
                    if amount <= 0:  # Ensure amount is greater than 0
                        page.open( ft.SnackBar(
                            ft.Text("Amount must be greater than 0.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                        ))
                    elif current_user:
                        if shares > amount:
                            users_db[current_user]["shares_owned"] -= amount
                            users_db[current_user]["portfolio_size"] += stock_price * amount
                            save_users()
                            page.open(ft.SnackBar(
                                ft.Text(f"Sale of {amount} shares successful!", color=ft.colors.WHITE),
                                bgcolor=ft.colors.GREEN
                            ))
                            show_home(page)  
                        else:
                            page.open(ft.SnackBar(
                                ft.Text("Insufficient shares, you cannot sell more shares than you own.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                        ))
                    else:
                        page.open(ft.SnackBar(
                            ft.Text("You must be logged in to sell shares.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                        ))
                except ValueError:  # Catch invalid numbers (e.g., letters, symbols)
                    page.open(ft.SnackBar(
                        ft.Text("Please enter a valid numeric amount.", color=ft.colors.WHITE), bgcolor=ft.colors.RED
                    ))
            
            page.snack_bar.open = True
            page.show_snack_bar()
            page.update()

        def sell_all():

            if current_user:
                users_db[current_user]["shares_owned"] -= shares
                users_db[current_user]["portfolio_size"] += stock_price * shares
                save_users()
                page.open(ft.SnackBar(
                    ft.Text(f"Sale of {shares} shares successful!", color=ft.colors.WHITE),
                    bgcolor=ft.colors.GREEN
                ))
                show_home(page) 

        
        deposit_field = ft.TextField(
            label="Number of Shares to sell",
            width=300,
            bgcolor=ft.colors.WHITE,
            keyboard_type=ft.KeyboardType.NUMBER,  # Restricts input to numbers on mobile
        )
        
        # Create portfolio breakdown container
        content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Sell Shares", size=40, weight="bold", color="#1e3a8a"),
                    ft.Text(f"Shared avaiable to sell: {shares}", color="#34495e"),
                    deposit_field,
                    ft.ElevatedButton(
                        text="Sell Shares",
                        on_click=lambda e: sell_shares(),
                        bgcolor="#1e3a8a",
                        color="#ffffff"
                    ),
                    ft.ElevatedButton(
                        text="Sell All Shares",
                        on_click=lambda e: sell_all(),
                        bgcolor="#1e3a8a",
                        color="#ffffff"
                    ),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,                
                alignment=ft.MainAxisAlignment.CENTER,
                spacing=20,
                scroll=ft.ScrollMode.AUTO,
            ),
            height=page.height,
            width=page.width,
            expand=True,  
            padding=ft.padding.all(20),
            bgcolor="#f4f6f9",
            gradient=gradient,
        )

        page.add(navbar, content)
        page.update()



    def show_contact(page):
        page.controls.clear()


        contact_content = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Contact Us", size=40, weight="bold", color="#1e3a8a"),
                    ft.Divider(),
                    ft.Text(
                        "To contact us please email us at ainvesting@notmail.com or call as at 0777755555",
                        size=20, color="#34495e"
                    ),
                    ft.Divider(),
                    ft.Text(
                        "Our team will be happy to assist you with any issues or inquires you may have involving:",
                        size=20, color="#34495e"
                    ),
                    ft.Text(f"• Why should I invest?", color="#34495e", size=15),
                    ft.Text(f"• Is investing safe?", color="#34495e", size=15),
                    ft.Text(f"• How do I deposit money?", color="#34495e", size=15),

                    
                ],
                alignment=ft.MainAxisAlignment.START,
                spacing=20,
                horizontal_alignment=ft.CrossAxisAlignment.START,
                scroll=ft.ScrollMode.AUTO,  
            ),
            height=page.height,
            width=page.width,
            expand=True,  
            padding=ft.padding.all(20),
            bgcolor="#ffffff",
            gradient=gradient,
        )


        page.add(ft.Column(
            controls=[
                navbar,  
                contact_content,
            ],
        ))
        page.update()

    def show_login(page):
        def handle_login(e):
            global current_user
            email = email_field.value
            password = password_field.value

            if email in users_db:
                stored_hashed_password = users_db[email]["password"]
                if bcrypt.checkpw(password.encode('utf-8'), stored_hashed_password.encode('utf-8')):
                    current_user = email
                    page.open(ft.SnackBar(ft.Text("Login successful!"), bgcolor=ft.colors.GREEN))
                    show_home(page)
                    return

            page.open(ft.SnackBar(ft.Text("Invalid email or password."), bgcolor=ft.colors.RED))
            page.snack_bar.open = True
            page.update()


        email_field = ft.TextField(label="Email", width=300, bgcolor=ft.colors.WHITE)
        password_field = ft.TextField(label="Password", password=True, width=300, bgcolor=ft.colors.WHITE)

        login_page = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Login", size=40, weight="bold", color="#1e3a8a"),
                    email_field,
                    password_field,
                    ft.ElevatedButton("Login", style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE), on_click=handle_login),
                    ft.TextButton("Don't have an account? Register", on_click=lambda _: show_register(page)),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=page.height,
            width=page.width,
            bgcolor="#ffffff",
            padding=ft.padding.all(20),
        )
        page.controls.clear()
        page.add(navbar, login_page)

    

    def show_register(page):
        def handle_register(e):
            email = email_field.value.lower()
            password = password_field.value
            confirm_password = confirm_password_field.value
            first_name = first_name_field.value
            last_name = last_name_field.value
            date_of_birth = dob_field.value

            email_regex = r"(^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$)"

            try:
                # Parse and validate the date of birth
                dob = datetime.strptime(date_of_birth, "%Y-%m-%d")
                age = (datetime.now() - dob).days // 365
            except ValueError:
                page.open(ft.SnackBar(ft.Text("Invalid Date of Birth format. Use YYYY-MM-DD."), bgcolor=ft.colors.RED))
                page.snack_bar.open = True
                page.update()
                return

            if not re.match(email_regex, email):
                page.open(ft.SnackBar(ft.Text("Invalid email address."), bgcolor=ft.colors.RED))
            elif age < 18:
                page.open(ft.SnackBar(ft.Text("You must be at least 18 years old to register."), bgcolor=ft.colors.RED))
            elif email in users_db:
                page.open(ft.SnackBar(ft.Text("Email already registered."), bgcolor=ft.colors.RED))
            elif password != confirm_password:
                page.open(ft.SnackBar(ft.Text("Passwords do not match."), bgcolor=ft.colors.RED))
            elif not first_name or not last_name:
                page.open(ft.SnackBar(ft.Text("First Name and Last Name cannot be empty."), bgcolor=ft.colors.RED))
            else:
                hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                users_db[email] = {
                    "password": hashed_password,
                    "name": f"{first_name} {last_name}",
                    "portfolio_size": 0.0,
                    "shares_owned": 0,
                    "first_name": first_name,
                    "last_name": last_name,
                    "dob": date_of_birth,
                }
                save_users()  # Save users to the JSON file
                page.open(ft.SnackBar(ft.Text("Registration successful!"), bgcolor=ft.colors.GREEN))
                show_login(page)

            page.snack_bar.open = True
            page.update()

        first_name_field = ft.TextField(label="First Name", width=300, bgcolor=ft.colors.WHITE)
        last_name_field = ft.TextField(label="Last Name", width=300, bgcolor=ft.colors.WHITE)
        email_field = ft.TextField(label="Email", width=300, bgcolor=ft.colors.WHITE)
        password_field = ft.TextField(label="Password", password=True, width=300, bgcolor=ft.colors.WHITE)
        confirm_password_field = ft.TextField(label="Confirm Password", password=True, width=300, bgcolor=ft.colors.WHITE)
        dob_field = ft.TextField(label="Date of Birth (YYYY-MM-DD)", width=300, bgcolor=ft.colors.WHITE)

        register_page = ft.Container(
            content=ft.Column(
                controls=[
                    ft.Text("Register", size=40, weight="bold", color="#1e3a8a"),
                    first_name_field,
                    last_name_field,
                    email_field,
                    password_field,
                    confirm_password_field,
                    dob_field,
                    ft.ElevatedButton("Register", style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE), on_click=handle_register),
                    ft.TextButton("Already have an account? Sign In", on_click=lambda _: show_login(page)),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            height=page.height,
            width=page.width,
            bgcolor="#ffffff",
            padding=ft.padding.all(20),
        )
        page.controls.clear()
        page.add(navbar, register_page)




    # Navigation bar with updated options
    navbar = ft.Container(
        content=ft.Row(
            controls=[
                ft.Text("AI Investing", size=28, weight="bold", color="#1e3a8a"),
                ft.Row(
                    controls=[
                        ft.TextButton("Home", style=ft.ButtonStyle(color="#1e3a8a"), on_click=lambda _: show_home(page)),
                        ft.TextButton("S&P 500",style=ft.ButtonStyle(color="#1e3a8a"), on_click=lambda _: show_sp500(page)),
                        ft.TextButton("Account", style=ft.ButtonStyle(color="#1e3a8a"), on_click=lambda _: show_account(page)),
                        ft.TextButton("About", style=ft.ButtonStyle(color="#1e3a8a"), on_click=lambda _: show_about(page)),
                        ft.TextButton("Contact", style=ft.ButtonStyle(color="#1e3a8a"), on_click=lambda _: show_contact(page)),
                        ft.PopupMenuButton(
                            items = [
                                ft.PopupMenuItem(text="About AI Investing", on_click=lambda _: show_about(page)),
                                ft.PopupMenuItem(text="More Info", on_click=lambda _: show_moreAbout(page)),
                                ft.PopupMenuItem(text="Glossary", on_click=lambda _: show_glossary(page)),
                            ],
                            # child = ft.TextButton("About", style=ft.ButtonStyle(color="#1e3a8a")),
                        ),
                        
                    ],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    spacing=20,
                ),
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
        ),
        padding=ft.padding.symmetric(horizontal=30, vertical=15),
        bgcolor="#e0f2f1",
        shadow=ft.BoxShadow(blur_radius=6, spread_radius=2, color=ft.colors.BLACK12),
    )


    
    home_section = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("Smart Investments with AI Investing", size=56, weight="bold", color="#1e3a8a"),
                ft.Text("Let AI make informed decisions and grow your wealth. Your financial future starts here.", size=20, color="#34495e"),
                ft.ElevatedButton(
                    "Get Started",
                    style=ft.ButtonStyle(bgcolor="#1e40af", color=ft.colors.WHITE),
                    on_click=lambda _: show_register(page),
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        height=page.height,
        width=page.width,
        expand=True,
        bgcolor="#ffffff",
        gradient=gradient,
        padding=ft.padding.all(20),
    )



    # Initialize with the home page
    page.add(navbar, home_section)

# Run the app
ft.app(target=main, assets_dir="assets", view=ft.AppView.WEB_BROWSER) # 
