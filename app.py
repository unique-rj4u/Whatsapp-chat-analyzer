# Importing modules
import nltk
import streamlit as st
from PIL import Image
import re
import preprocessor,helper
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use('dark_background')


hide_streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            footer:after {
                content:'Made with ❤ by Dhiraj Sahani'; 
                visibility: visible;
                display: block;
                position: relative;
                #background-color: red;
                padding: 5px;
                top: 2px;
            }
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#Center the sidebar whatsapp image:
st.markdown(
    """
    <style>
        [data-testid=stSidebar] [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%; 
        }

        p {
        background-image: url(‘img_file.jpg’);
        }
        
    </style>
    """, unsafe_allow_html=True
)



with st.sidebar:
    st.image("whatsapp.png", width=110)

# Title of the sidebar
st.sidebar.title("WhatsApp Chat Analyzer")



# VADER : is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments.
nltk.download('vader_lexicon')

# File upload button
uploaded_file = st.sidebar.file_uploader("Choose a file")


if uploaded_file is not None:
    
    # Getting byte form & then decoding
    bytes_data = uploaded_file.getvalue()
    d = bytes_data.decode("utf-8")
    
    # Perform preprocessing
    df = preprocessor.preprocess(d)
    
    # select option:
    user_menu = st.sidebar.radio(
        'Select an Option',
        ('Basic Analysis','Sentiment Analysis', 'Both')
    )
    

#-----------------------------------------------------------------<< Basic Analysis >>------------------------------------------------------------------#

    if user_menu == 'Basic Analysis':

        #fetch unique users
        user_list = df['user'].unique().tolist()
        #removing the group_notification:
        # user_list.remove('group_notification')
        user_list.sort()
        #inserting the Overall selection:
        user_list.insert(0,"Overall")
        
        selected_user = st.sidebar.selectbox("Select User:", user_list)

        if st.sidebar.button("Show Analysis"):


            # Main heading
            st. markdown("<h1 style='text-align: center; color: green;'>Basic Analysis</h1>", unsafe_allow_html=True)

            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            
            st.title("Top Statistics:")

            col1, col2, col3, col4 = st.columns(4)
            

            with col1:
                st.header("Total Messages:")
                st.title(num_messages)

            with col2:
                st.header("Total Words:")
                st.title(words)

            with col3:
                st.header("Media Shared:")
                st.title(num_media_messages)

            with col4:
                st.header("Links Shared:")
                st.title(num_links)

            # Monthly timeline:
            st.title("Monthly timeline:")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color = '#780413')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Daily timeline:
            st.title("Daily timeline:")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color = '#780413')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # activity map:
            st.title("Activity map:")
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most busy day:")
                busy_day = helper.week_activity_map(selected_user, df)
                fig,ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color = '#780413')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most busy month:")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color = '#780413')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            st.title("Weekly Activity Map:")
            user_heatmap = helper.activity_heatmap(selected_user,df)
            fig,ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)



            #finding the busiest users in the group (group level)
            if selected_user == 'Overall':
                st.title('Most Busy Users:')
                x, new_df = helper.most_busy_users(df)
                fig, ax = plt.subplots()
                
                col1, col2 = st.columns(2)


                with col1:
                    ax.bar(x.index, x.values, color = '#780413')
                    plt.xticks(rotation = 'vertical')
                    st.pyplot(fig)

                with col2:
                    st.dataframe(new_df)

            # WordCloud:
            st.title('WordCloud:')
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            # most common words:
            most_common_df = helper.most_common_words(selected_user, df)

            fig, ax = plt.subplots()

            ax.barh(most_common_df[0], most_common_df[1])

            plt.xticks(rotation = 'vertical')

            st.title('Most common words:')
            st.pyplot(fig)



            # emoji analysis:
            emoji_df = helper.emoji_helper(selected_user, df)

            st.title("Emoji Analysis:")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)

            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)


#----------------------------------------------------------------<< Sentiment Analysis >>---------------------------------------------------------------#

    if user_menu == 'Sentiment Analysis':

        # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        
        # Object
        sentiments = SentimentIntensityAnalyzer()
        
        # Creating different columns for (Positive/Negative/Neutral)
        df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]] # Positive
        df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]] # Negative
        df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]] # Neutral
        
        # To identify true sentiment per row in message column
        def sentiment(d):
            if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
                return 1
            if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
                return -1
            if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
                return 0

        # Creating new column & Applying function
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)
        
        # User names list
        user_list = df['user'].unique().tolist()

        #removing the group_notification:
        # user_list.remove('group_notification')
        
        # Sorting
        user_list.sort()
        
        # Insert "Overall" at index 0
        user_list.insert(0, "Overall")

        
        
        # Selectbox
        selected_user = st.sidebar.selectbox("Select User:", user_list)

        if st.sidebar.button("Show Analysis"):


            # Main heading
            st. markdown("<h1 style='text-align: center; color: green;'>Sentiment Analysis</h1>", unsafe_allow_html=True)


            # Monthly activity map
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Positive)</h3>",unsafe_allow_html=True)
                
                busy_month = helper.month_activity_map_2(selected_user, df,1)
                
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Neutral)</h3>",unsafe_allow_html=True)
                
                busy_month = helper.month_activity_map_2(selected_user, df, 0)
                
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Monthly Activity map(Negative)</h3>",unsafe_allow_html=True)
                
                busy_month = helper.month_activity_map_2(selected_user, df, -1)
                
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Daily activity map
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Positive)</h3>",unsafe_allow_html=True)
                
                busy_day = helper.week_activity_map_2(selected_user, df,1)
                
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Neutral)</h3>",unsafe_allow_html=True)
                
                busy_day = helper.week_activity_map_2(selected_user, df, 0)
                
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Daily Activity map(Negative)</h3>",unsafe_allow_html=True)
                
                busy_day = helper.week_activity_map_2(selected_user, df, -1)
                
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Weekly activity map
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Positive)</h3>",unsafe_allow_html=True)
                    
                    user_heatmap = helper.activity_heatmap_2(selected_user, df, 1)
                    
                    fig, ax = plt.subplots()
                    ax = sns.heatmap(user_heatmap)
                    st.pyplot(fig)
                except:
                    st.image('error.webp')
            with col2:
                try:
                    st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Neutral)</h3>",unsafe_allow_html=True)
                    
                    user_heatmap = helper.activity_heatmap_2(selected_user, df, 0)
                    
                    fig, ax = plt.subplots()
                    ax = sns.heatmap(user_heatmap)
                    st.pyplot(fig)
                except:
                    st.image('error.webp')
            with col3:
                try:
                    st.markdown("<h3 style='text-align: center; color: black;'>Weekly Activity Map(Negative)</h3>",unsafe_allow_html=True)
                    
                    user_heatmap = helper.activity_heatmap_2(selected_user, df, -1)
                    
                    fig, ax = plt.subplots()
                    ax = sns.heatmap(user_heatmap)
                    st.pyplot(fig)
                except:
                    st.image('error.webp')

            # Daily timeline
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Positive)</h3>",unsafe_allow_html=True)
                
                daily_timeline = helper.daily_timeline_2(selected_user, df, 1)
                
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Neutral)</h3>",unsafe_allow_html=True)
                
                daily_timeline = helper.daily_timeline_2(selected_user, df, 0)
                
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Daily Timeline(Negative)</h3>",unsafe_allow_html=True)
                
                daily_timeline = helper.daily_timeline_2(selected_user, df, -1)
                
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Monthly timeline
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Positive)</h3>",unsafe_allow_html=True)
                
                timeline = helper.monthly_timeline_2(selected_user, df,1)
                
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Neutral)</h3>",unsafe_allow_html=True)
                
                timeline = helper.monthly_timeline_2(selected_user, df,0)
                
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: black;'>Monthly Timeline(Negative)</h3>",unsafe_allow_html=True)
                
                timeline = helper.monthly_timeline_2(selected_user, df,-1)
                
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Percentage contributed
            if selected_user == 'Overall':
                col1,col2,col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Contribution</h3>",unsafe_allow_html=True)
                    x = helper.percentage(df, 1)
                    
                    # Displaying
                    st.dataframe(x)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Contribution</h3>",unsafe_allow_html=True)
                    y = helper.percentage(df, 0)
                    
                    # Displaying
                    st.dataframe(y)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Contribution</h3>",unsafe_allow_html=True)
                    z = helper.percentage(df, -1)
                    
                    # Displaying
                    st.dataframe(z)


            # Most Positive,Negative,Neutral User...
            if selected_user == 'Overall':
                
                # Getting names per sentiment
                x = df['user'][df['value'] == 1].value_counts().head(10)
                y = df['user'][df['value'] == -1].value_counts().head(10)
                z = df['user'][df['value'] == 0].value_counts().head(10)

                col1,col2,col3 = st.columns(3)
                with col1:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;'>Most Positive Users</h3>",unsafe_allow_html=True)
                    
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values, color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;'>Most Neutral Users</h3>",unsafe_allow_html=True)
                    
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(z.index, z.values, color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col3:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;'>Most Negative Users</h3>",unsafe_allow_html=True)
                    
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(y.index, y.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

            # WORDCLOUD......
            col1,col2,col3 = st.columns(3)
            with col1:
                try:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;'>Positive WordCloud</h3>",unsafe_allow_html=True)
                    
                    # Creating wordcloud of positive words
                    df_wc = helper.create_wordcloud_2(selected_user, df,1)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)
                except:
                    # Display error message
                    st.image('error.webp')
            with col2:
                try:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;'>Neutral WordCloud</h3>",unsafe_allow_html=True)
                    
                    # Creating wordcloud of neutral words
                    df_wc = helper.create_wordcloud_2(selected_user, df,0)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)
                except:
                    # Display error message
                    st.image('error.webp')
            with col3:
                try:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;'>Negative WordCloud</h3>",unsafe_allow_html=True)
                    
                    # Creating wordcloud of negative words
                    df_wc = helper.create_wordcloud_2(selected_user, df,-1)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)
                except:
                    # Display error message
                    st.image('error.webp')

            # Most common positive words
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    # Data frame of most common positive words.
                    most_common_df = helper.most_common_words_2(selected_user, df,1)
                    
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;'>Positive Words</h3>",unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df[0], most_common_df[1],color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                except:
                    # Disply error image
                    st.image('error.webp')
            with col2:
                try:
                    # Data frame of most common neutral words.
                    most_common_df = helper.most_common_words_2(selected_user, df,0)
                    
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;'>Neutral Words</h3>",unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df[0], most_common_df[1],color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                except:
                    # Disply error image
                    st.image('error.webp')
            with col3:
                try:
                    # Data frame of most common negative words.
                    most_common_df = helper.most_common_words_2(selected_user, df,-1)
                    
                    # heading
                    st.markdown("<h3 style='text-align: center; color: black;'>Negative Words</h3>",unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df[0], most_common_df[1], color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                except:
                    # Disply error image
                    st.image('error.webp')


#---------------------------------------------------------------------<< Both >>------------------------------------------------------------------------#

    if user_menu == 'Both':

         # Importing SentimentIntensityAnalyzer class from "nltk.sentiment.vader"
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        
        # Object
        sentiments = SentimentIntensityAnalyzer()
        
        # Creating different columns for (Positive/Negative/Neutral)
        df["po"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]] # Positive
        df["ne"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]] # Negative
        df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]] # Neutral
        
        # To identify true sentiment per row in message column
        def sentiment(d):
            if d["po"] >= d["ne"] and d["po"] >= d["nu"]:
                return 1
            if d["ne"] >= d["po"] and d["ne"] >= d["nu"]:
                return -1
            if d["nu"] >= d["po"] and d["nu"] >= d["ne"]:
                return 0

        # Creating new column & Applying function
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)
        
        # User names list
        user_list = df['user'].unique().tolist()

        #removing the group_notification:
        # user_list.remove('group_notification')
        
        # Sorting
        user_list.sort()
        
        # Insert "Overall" at index 0
        user_list.insert(0, "Overall")
        
        selected_user = st.sidebar.selectbox("Select User:", user_list)

        if st.sidebar.button("Show Analysis"):


            # Main heading
            st. markdown("<h1 style='text-align: center; color: green;'>Basic Analysis</h1>", unsafe_allow_html=True)

            

            num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
            
            st.title("Top Statistics:")

            col1, col2, col3, col4 = st.columns(4)
            

            with col1:
                st.header("Total Messages:")
                st.title(num_messages)

            with col2:
                st.header("Total Words:")
                st.title(words)

            with col3:
                st.header("Media Shared:")
                st.title(num_media_messages)

            with col4:
                st.header("Links Shared:")
                st.title(num_links)

            # Monthly timeline:
            st.title("Monthly timeline:")
            timeline = helper.monthly_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(timeline['time'], timeline['message'], color = '#780413')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # Daily timeline:
            st.title("Daily timeline:")
            daily_timeline = helper.daily_timeline(selected_user, df)
            fig, ax = plt.subplots()
            ax.plot(daily_timeline['only_date'], daily_timeline['message'], color = '#780413')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

            # activity map:
            st.title("Activity map:")
            col1, col2 = st.columns(2)

            with col1:
                st.header("Most busy day:")
                busy_day = helper.week_activity_map(selected_user, df)
                fig,ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color = '#780413')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            with col2:
                st.header("Most busy month:")
                busy_month = helper.month_activity_map(selected_user, df)
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color = '#780413')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            st.title("Weekly Activity Map:")
            user_heatmap = helper.activity_heatmap(selected_user,df)
            fig,ax = plt.subplots()
            ax = sns.heatmap(user_heatmap)
            st.pyplot(fig)



            #finding the busiest users in the group (group level)
            if selected_user == 'Overall':
                st.title('Most Busy Users:')
                x, new_df = helper.most_busy_users(df)
                fig, ax = plt.subplots()
                
                col1, col2 = st.columns(2)


                with col1:
                    ax.bar(x.index, x.values, color = '#780413')
                    plt.xticks(rotation = 'vertical')
                    st.pyplot(fig)

                with col2:
                    st.dataframe(new_df)

            # WordCloud:
            st.title('WordCloud:')
            df_wc = helper.create_wordcloud(selected_user, df)
            fig, ax = plt.subplots()
            ax.imshow(df_wc)
            st.pyplot(fig)

            # most common words:
            most_common_df = helper.most_common_words(selected_user, df)

            fig, ax = plt.subplots()

            ax.barh(most_common_df[0], most_common_df[1])

            plt.xticks(rotation = 'vertical')

            st.title('Most common words:')
            st.pyplot(fig)



            # emoji analysis:
            emoji_df = helper.emoji_helper(selected_user, df)

            st.title("Emoji Analysis:")

            col1, col2 = st.columns(2)

            with col1:
                st.dataframe(emoji_df)

            with col2:
                fig, ax = plt.subplots()
                ax.pie(emoji_df[1].head(), labels=emoji_df[0].head(), autopct="%0.2f")
                st.pyplot(fig)


            st.write("---")

            # Main heading
            st. markdown("<h1 style='text-align: center; color: green;'>Sentiment Analysis</h1>", unsafe_allow_html=True)


            # Monthly activity map
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: White;'>Monthly Activity map(Positive)</h3>",unsafe_allow_html=True)
                
                busy_month = helper.month_activity_map_2(selected_user, df,1)
                
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: White;'>Monthly Activity map(Neutral)</h3>",unsafe_allow_html=True)
                
                busy_month = helper.month_activity_map_2(selected_user, df, 0)
                
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: White;'>Monthly Activity map(Negative)</h3>",unsafe_allow_html=True)
                
                busy_month = helper.month_activity_map_2(selected_user, df, -1)
                
                fig, ax = plt.subplots()
                ax.bar(busy_month.index, busy_month.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Daily activity map
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: White;'>Daily Activity map(Positive)</h3>",unsafe_allow_html=True)
                
                busy_day = helper.week_activity_map_2(selected_user, df,1)
                
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: White;'>Daily Activity map(Neutral)</h3>",unsafe_allow_html=True)
                
                busy_day = helper.week_activity_map_2(selected_user, df, 0)
                
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: White;'>Daily Activity map(Negative)</h3>",unsafe_allow_html=True)
                
                busy_day = helper.week_activity_map_2(selected_user, df, -1)
                
                fig, ax = plt.subplots()
                ax.bar(busy_day.index, busy_day.values, color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Weekly activity map
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    st.markdown("<h3 style='text-align: center; color: White;'>Weekly Activity Map(Positive)</h3>",unsafe_allow_html=True)
                    
                    user_heatmap = helper.activity_heatmap_2(selected_user, df, 1)
                    
                    fig, ax = plt.subplots()
                    ax = sns.heatmap(user_heatmap)
                    st.pyplot(fig)
                except:
                    st.image('error.webp')
            with col2:
                try:
                    st.markdown("<h3 style='text-align: center; color: White;'>Weekly Activity Map(Neutral)</h3>",unsafe_allow_html=True)
                    
                    user_heatmap = helper.activity_heatmap_2(selected_user, df, 0)
                    
                    fig, ax = plt.subplots()
                    ax = sns.heatmap(user_heatmap)
                    st.pyplot(fig)
                except:
                    st.image('error.webp')
            with col3:
                try:
                    st.markdown("<h3 style='text-align: center; color: White;'>Weekly Activity Map(Negative)</h3>",unsafe_allow_html=True)
                    
                    user_heatmap = helper.activity_heatmap_2(selected_user, df, -1)
                    
                    fig, ax = plt.subplots()
                    ax = sns.heatmap(user_heatmap)
                    st.pyplot(fig)
                except:
                    st.image('error.webp')

            # Daily timeline
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: White;'>Daily Timeline(Positive)</h3>",unsafe_allow_html=True)
                
                daily_timeline = helper.daily_timeline_2(selected_user, df, 1)
                
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: White;'>Daily Timeline(Neutral)</h3>",unsafe_allow_html=True)
                
                daily_timeline = helper.daily_timeline_2(selected_user, df, 0)
                
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: White;'>Daily Timeline(Negative)</h3>",unsafe_allow_html=True)
                
                daily_timeline = helper.daily_timeline_2(selected_user, df, -1)
                
                fig, ax = plt.subplots()
                ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Monthly timeline
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("<h3 style='text-align: center; color: White;'>Monthly Timeline(Positive)</h3>",unsafe_allow_html=True)
                
                timeline = helper.monthly_timeline_2(selected_user, df,1)
                
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='green')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.markdown("<h3 style='text-align: center; color: White;'>Monthly Timeline(Neutral)</h3>",unsafe_allow_html=True)
                
                timeline = helper.monthly_timeline_2(selected_user, df,0)
                
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='grey')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col3:
                st.markdown("<h3 style='text-align: center; color: White;'>Monthly Timeline(Negative)</h3>",unsafe_allow_html=True)
                
                timeline = helper.monthly_timeline_2(selected_user, df,-1)
                
                fig, ax = plt.subplots()
                ax.plot(timeline['time'], timeline['message'], color='red')
                plt.xticks(rotation='vertical')
                st.pyplot(fig)

            # Percentage contributed
            if selected_user == 'Overall':
                col1,col2,col3 = st.columns(3)
                with col1:
                    st.markdown("<h3 style='text-align: center; color: White;'>Most Positive Contribution</h3>",unsafe_allow_html=True)
                    x = helper.percentage(df, 1)
                    
                    # Displaying
                    st.dataframe(x)
                with col2:
                    st.markdown("<h3 style='text-align: center; color: White;'>Most Neutral Contribution</h3>",unsafe_allow_html=True)
                    y = helper.percentage(df, 0)
                    
                    # Displaying
                    st.dataframe(y)
                with col3:
                    st.markdown("<h3 style='text-align: center; color: White;'>Most Negative Contribution</h3>",unsafe_allow_html=True)
                    z = helper.percentage(df, -1)
                    
                    # Displaying
                    st.dataframe(z)


            # Most Positive,Negative,Neutral User...
            if selected_user == 'Overall':
                
                # Getting names per sentiment
                x = df['user'][df['value'] == 1].value_counts().head(10)
                y = df['user'][df['value'] == -1].value_counts().head(10)
                z = df['user'][df['value'] == 0].value_counts().head(10)

                col1,col2,col3 = st.columns(3)
                with col1:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: White;'>Most Positive Users</h3>",unsafe_allow_html=True)
                    
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(x.index, x.values, color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col2:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: White;'>Most Neutral Users</h3>",unsafe_allow_html=True)
                    
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(z.index, z.values, color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                with col3:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: White;'>Most Negative Users</h3>",unsafe_allow_html=True)
                    
                    # Displaying
                    fig, ax = plt.subplots()
                    ax.bar(y.index, y.values, color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)

            # WORDCLOUD......
            col1,col2,col3 = st.columns(3)
            with col1:
                try:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: White;'>Positive WordCloud</h3>",unsafe_allow_html=True)
                    
                    # Creating wordcloud of positive words
                    df_wc = helper.create_wordcloud_2(selected_user, df,1)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)
                except:
                    # Display error message
                    st.image('error.webp')
            with col2:
                try:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: White;'>Neutral WordCloud</h3>",unsafe_allow_html=True)
                    
                    # Creating wordcloud of neutral words
                    df_wc = helper.create_wordcloud_2(selected_user, df,0)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)
                except:
                    # Display error message
                    st.image('error.webp')
            with col3:
                try:
                    # heading
                    st.markdown("<h3 style='text-align: center; color: White;'>Negative WordCloud</h3>",unsafe_allow_html=True)
                    
                    # Creating wordcloud of negative words
                    df_wc = helper.create_wordcloud_2(selected_user, df,-1)
                    fig, ax = plt.subplots()
                    ax.imshow(df_wc)
                    st.pyplot(fig)
                except:
                    # Display error message
                    st.image('error.webp')

            # Most common positive words
            col1, col2, col3 = st.columns(3)
            with col1:
                try:
                    # Data frame of most common positive words.
                    most_common_df = helper.most_common_words_2(selected_user, df,1)
                    
                    # heading
                    st.markdown("<h3 style='text-align: center; color: White;'>Positive Words</h3>",unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df[0], most_common_df[1],color='green')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                except:
                    # Disply error image
                    st.image('error.webp')
            with col2:
                try:
                    # Data frame of most common neutral words.
                    most_common_df = helper.most_common_words_2(selected_user, df,0)
                    
                    # heading
                    st.markdown("<h3 style='text-align: center; color: White;'>Neutral Words</h3>",unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df[0], most_common_df[1],color='grey')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                except:
                    # Disply error image
                    st.image('error.webp')
            with col3:
                try:
                    # Data frame of most common negative words.
                    most_common_df = helper.most_common_words_2(selected_user, df,-1)
                    
                    # heading
                    st.markdown("<h3 style='text-align: center; color: White;'>Negative Words</h3>",unsafe_allow_html=True)
                    fig, ax = plt.subplots()
                    ax.barh(most_common_df[0], most_common_df[1], color='red')
                    plt.xticks(rotation='vertical')
                    st.pyplot(fig)
                except:
                    # Disply error image
                    st.image('error.webp')


else:

    # Main heading
    st. markdown("<h1 style='text-align: center; color: green;'>WhatsApp Chat Analysis</h1>", unsafe_allow_html=True)

    st. markdown("<h2 style='text-align: left; color: White;'>How to export chat: </h2>", unsafe_allow_html=True)
    image = Image.open('Export.png')

    st.image(image)

    st. markdown("<h2 style='text-align: left; color: White;'>Steps:</h3>", unsafe_allow_html=True)

    st. markdown("<h3 style='text-align: left; color: yellow;'>⚠  Do Export in 24 hours format</h3>", unsafe_allow_html=True)
    st. markdown('Step-1: Tap on the  "⋮ (three dots)" at the top of the screen')
    st. markdown('Step-2: Select  "More"')
    st. markdown('Step-3: Tap on  "Export chat"')
    st. markdown('Step-4: Select  "WITHOUT MEDIA" ')