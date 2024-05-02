import streamlit as st
from fetchExtractArticles import extract_text, fetch_articles
from summaryGeneration import generate_summary
from suggestedArticles import get_suggestions

# import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


# Download NLTK resources (if not already downloaded)
# nltk.download('punkt')


SEARCH_RESULTS_START = 1
SEARCH_RESULTS_STOP = 5


def main():
    st.set_page_config(layout="wide")
    st.header("Quick Search")

    input_query = st.text_input("Enter your search query:")
    max_token_limit = st.number_input(
        "Select max token limit for summary",
        min_value=50,
        max_value=1000,
        value=200,
        step=10,
    )

    # Add a hidden button to trigger the function
    # when the Enter key is pressed
    #     st.markdown("""
    #     <button id="Search" style="display: none;" type="button" onclick="document.dispatchEvent(new Event('custom-event'));"></button>
    #     <script>
    #         document.addEventListener('keypress', function(event) {
    #             if (event.key === 'Enter') {
    #                 document.getElementById('Search').click();
    #             }
    #         });
    #     </script>
    # """, unsafe_allow_html=True)

    if st.button("Search"):
        articles = fetch_articles(
            input_query, start=SEARCH_RESULTS_START, stop=SEARCH_RESULTS_STOP
        )
        articles = articles[:4]
        full_content = ""
        st.write("Sources:")
        for article in articles:
            try:
                result = extract_text(article)
                st.write(result["url"])
                full_content += result["content"]
                st.subheader(result["title"])
                st.write(result["content"])
                st.markdown("---")
            except Exception as e:
                # st.error(f"Error processing article: {e}")
                continue
        # st.write(full_content)
        # tokens = word_tokenize(full_content)
        # Remove stopwords
        # stop_words = set(stopwords.words("english"))
        # filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        # filtered_text = " ".join(filtered_tokens)
        summary = generate_summary(full_content, max_token_limit)
        # summary = generate_summary(filtered_text,max_token_limit)
        st.write(len(word_tokenize(summary)))

        st.subheader("Summary:")
        st.write(summary)

        st.subheader("Articles you may be interested in")
        urls = get_suggestions(user_query=input_query, summary=summary)
        for i, (k, v) in enumerate(urls.items()):
            st.write("Query + Keyword: " + k)
            if i == 3:
                break
            st.write(v[0])


if __name__ == "__main__":
    main()
