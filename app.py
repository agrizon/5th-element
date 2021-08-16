import streamlit as st


def main():
    menu = ["Home", "Search", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == 'Home':
        st.subheader('Home')
        st.success('Full Layout')
        # st.text('Sidebar')

        col1,  col2 = st.columns(2)

        with col1:
            col1.success('First Column')
            search = st.text_input("Enter text for search:")
            if st.button('Submit'):
                st.write(search.upper())

        with col2:
            col2.success('Second Column')
            year = st.number_input('Year:', 1995, 2021)

    elif choice == 'Search':
        st.subheader('Search')
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("From column 1")
        with col2:
            st.info("From Column 2")
    else:
        st.subheader('About')


if __name__ == '__main__':
    main()
