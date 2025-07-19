# -*- coding: utf-8 -*-
"""
Created on Thu Jul 17 07:24:59 2025

@author: win11
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
from itertools import chain
import numpy as np

vizro_bootstrap = "https://cdn.jsdelivr.net/gh/mckinsey/vizro@main/vizro-core/src/vizro/static/css/vizro-bootstrap.min.css?v=2"


# Load data
books_df = pd.read_csv("goodreads_works_v1.csv")
reviews_df = pd.read_csv("goodreads_reviews.csv", low_memory=False)

# Merge reviews into books
reviews_grouped = reviews_df.groupby("work_id")["review_text"].apply(lambda texts: " ".join(str(t) for t in texts)).reset_index()
books_df = books_df.merge(reviews_grouped, on="work_id", how="left")

# Convert genre strings to lists and collect unique genres
books_df["genre_list"] = books_df["genres"].fillna("").apply(lambda g: [genre.strip() for genre in g.split(",") if genre.strip()])
unique_genres = sorted(set(chain.from_iterable(books_df["genre_list"])))

# -----------------------
# ✳️ Normalize & prioritize fields
# -----------------------
books_df["original_title_lower"] = books_df["original_title"].fillna("").str.lower()
books_df["author_lower"] = books_df["author"].fillna("").str.lower()
books_df["genres_lower"] = books_df["genres"].fillna("").str.lower()
books_df["description_lower"] = books_df["description"].fillna("").str.lower()
books_df["review_text_lower"] = books_df["review_text"].fillna("").str.lower()

# Boost important fields like author and genres (using lowercase versions)
books_df["text"] = (
    (books_df["original_title_lower"] + " ") +
    (books_df["genres_lower"] + " ") * 2 +
    books_df["description_lower"] + " " +
    (books_df["author_lower"] + " ") * 4 +
    books_df["review_text_lower"]
)

# -----------------------
# ✳️ TF-IDF
# -----------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(books_df["text"])

# -----------------------
# Dash App with Bootstrap
# -----------------------
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app = dash.Dash(__name__, external_stylesheets=[vizro_bootstrap, dbc.icons.FONT_AWESOME])
# Add custom favicon
app._favicon = "logo.png"

app.title = "Create your reading list"



# Theme toggle component
# theme_toggle = html.Div([
#     dbc.Label(className="fa fa-moon", html_for="theme-switch"),
#     dbc.Switch(id="theme-switch", value=False, className="d-inline-block ms-1", persistence=True),
#     dbc.Label(className="fa fa-sun", html_for="theme-switch"),
# ], className="d-flex align-items-center")

# Main layout
app.layout = dbc.Container([
    dcc.Store(id="theme-store", data=False),
    # Add these after your dcc.Store(id="theme-store", data=False) line:
    dcc.Store(id="loved-books-store", data=[], storage_type="local"),  # Persists across sessions
    dcc.Download(id="download-loves"),
    
dbc.Row([
    dbc.Col([
        html.Div([
            html.Img(src="/assets/logo.png", height="50px", className="me-3"),
            html.H1("Next read?", className="d-inline-block mb-0"),
        ], className="d-flex align-items-center")
    ], className="col-12 col-md-9 order-2 order-md-1"),  # Reduced from 9 to make room
    dbc.Col([
        dbc.Button(
            [html.I(className="fa fa-download me-2"), "Favorites"], 
            id="export-loves-btn", 
            color="primary", 
            size="sm",
            className="me-3"
        ),
        dbc.Button(
            html.I(className="fa fa-question"),
            id="help-button",
            color="primary",
            #outline=True,
            size="sm",
            className="me-3"
        ),
        #theme_toggle
        dbc.Label(className="fa fa-moon", html_for="theme-switch"),
        dbc.Switch(id="theme-switch", value=False, className="d-inline-block ms-1 me-1", persistence=True),
        dbc.Label(className="fa fa-sun", html_for="theme-switch"),
        
        
    ], className="col-12 col-md-3 order-1 order-md-2 d-flex align-items-center justify-content-end"),

], className="mt-3 flex-align-items-center", style={'marginBottom':'2rem'}),
    
    dbc.Row([
        dbc.Col([
            dbc.InputGroup([
                dbc.Input(
                    id="query-input", 
                    placeholder="Enter keywords, author name or title ...", 
                    size="lg", 
                    debounce=True  # This triggers callback on Enter or after pause
                    ),
                
            dbc.Button(
                html.I(className="fa fa-times"),
                id="clear-button",
                color="secondary",
                outline=True,
                size="lg"
            ),
            dbc.Button(html.I(className="fa fa-search me-2"), id="search-button", color="primary", n_clicks=0, size="lg")
                ], className="mb-3")
        ], className="col-12 col-md-6"),
   # ]),
    
   # dbc.Row([
        dbc.Col([
            dbc.Select(
                id="genre-filter",
                options=[{"label": "All genres", "value": "All"}] + [{"label": genre, "value": genre} for genre in unique_genres],
                value="All",  # Set default value
                placeholder="Filter by genre...",
                className="mb-4"
            )
        ], className="col-12 col-md-6")
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id="results-container")
        ],  className="col-12")
    ]),
    
    html.Hr(className="mt-5"),  # Horizontal line separator
    dbc.Row([
        dbc.Col([
            html.Footer([
                "by ",
                dbc.NavLink(
                    "Marie-Anne Melis",
                    href="https://www.linkedin.com/in/marieannemelis/",
                    target="_blank",  # Opens in new tab
                    className="text-decoration-none"
                )
            ], className="text-center text-muted py-3")
        ],  className="col-12")
    ], className="mt-auto"),
    dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("", id="modal-title")),
        dbc.ModalBody(id="modal-body"),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-modal", className="ms-auto", n_clicks=0)
        ),
    ],
    id="book-details-modal",
    size="xl",
    scrollable=True,
    is_open=False,
),
    dbc.Modal(
    [
        dbc.ModalHeader(
            dbc.ModalTitle(html.H3('Help')),
            close_button=True  # This adds the X in upper right
        ),
        dbc.ModalBody([
            #Search
            html.H3([html.I(className="fa fa-book"),' Searching and saving books'], style={'marginTop':'2rem'}),
            dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                dcc.Markdown('''
                   If you enter one or more keywords this is happening in this order:
                       
                   * Author: if you enter an authorname or a string that is part of an authorname,
                   you will get a list of the authors books in return. The result is ordered by
                   relevance.
                   * Title: if your keyword(s) match part or a booktitle, those books will be
                   returned. The result is ordered by most recent book first.
                   * If both search attempts give an empty result, the system uses intelligence
                   to find the most appropriate books for your query. The result is ordered by relevance.
                   
                   
                   Warning: if you search for example for an author in a genre which was not assigned
                   to one of the authors books, you will get a strange result.
                   See it as an opportunity to find hidden gems.

            '''),
                
            ],
            title="Order of results",
        ),
        
        dbc.AccordionItem(
            dcc.Markdown('''
               Use genres as a startingpoint for your search or to filter down
               searchresults.
               * Startingpoint: select a genre without keywords
               * Filter: select a combination of genre and keyword(s)
               
               Warning: most books are assigned to many genres.

        '''),
            title="Genres",
        ),
            dbc.AccordionItem(
                dcc.Markdown('''
                  When you press the download button, the system will start the
                  download off all your loved books in the form of a .csv file.
                  You can store this file on your computer, it contains author
                  and title.

            '''),
                title="Download favorites",
            ),
        ],start_collapsed=True,
        ),
         #Search
         html.H3([html.I(className="fa fa-bug"),' Bugs and remarks'], style={'marginTop':'2rem'}),
         dbc.Accordion(
 [
     dbc.AccordionItem(
         [
             dcc.Markdown('''
                Known bugs:
                    
                * sometimes the heart button does not work. This has to do with a lack of datacleaning.
                * sometimes when you click on an author name, nothing happens. This has to do with a lack of datacleaning.
                    
                Sorry, this is a prototype.

         '''),
             
         ],
         title="Bugs",
     ),
     
     dbc.AccordionItem(
         dcc.Markdown('''
            You see a download button because the creator of this app does not want to sponsor people
            who are already rich.
            
            Download the list, shop the book at your local bookstore or borrow it in your local
            library.

     '''),
         title="Why a download icon and not a shop button",
     ),
         dbc.AccordionItem(
             dcc.Markdown('''
                Possible improvements list:          
                          
                * back button going to previous query result
                * number of results for query
                * test a bit more with NLP
                

         '''),
             title="Improvements",
         ),
     ],start_collapsed=True,
     )
        ]),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-help-modal", className="ms-auto", n_clicks=0)
        ),
    ],
    id="help-modal",
    size="lg",
    scrollable=True,
    is_open=False,
),
], fluid=False, id="main-container")

# -----------------------
# Helper function to create book card
# -----------------------
def create_book_card(book, loved_books):
    # Check if image_url exists, otherwise use placeholder
    image_url = book.get('image_url', f"https://via.placeholder.com/120x180.png?text={book['original_title'][:10]}...")
    
    # Check if this book is loved
    book_id = f"{book['original_title']}_{book['author']}"  # Create unique ID
    
    # Check if book is in loved_books (handle both dict and string formats)
    is_loved = False
    for loved_book in loved_books:
        if isinstance(loved_book, dict):
            if loved_book.get('id') == book_id:
                is_loved = True
                break
        elif loved_book == book_id:
            is_loved = True
            break
    
    card = dbc.Card([
        dbc.Row([
            dbc.Col([
                dbc.CardImg(
                    src=image_url,
                    className="img-fluid rounded-start",
                    style={"width": "180px", "objectFit": "cover"}
                )
            ], className="col-12 col-md-2"),
            dbc.Col([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H2(book['original_title'], className="card-title mb-2"),
                        ], width=10),
                        dbc.Col([
                            dbc.Button(
                                html.I(className=f"fa {'fa-heart' if is_loved else 'fa-heart'}"),
                                id={"type": "love-button", "index": book_id},
                                color="danger" if is_loved else "secondary",
                                outline=not is_loved,
                                size="lg",
                                className="float-end"
                            )
                        ], width=2)
                    ]),
                    html.H3([
                        "by ",
                        dbc.Button(
                            book['author'],
                            id={"type": "author-link", "index": book['author']},
                            color="link",
                            className="p-0 text-muted text-decoration-none inline-author-link",
                            style={"fontSize": "inherit", "fontWeight": "inherit"}
                        )
                    ], className="text-muted mb-3"),
                    html.P([
                        dbc.Badge(f"{int(book['original_publication_year'])}", color="primary", className="me-2 mt-2"),
                        dbc.Badge(f"★ {book['avg_rating']}", color="warning", className="me-2 mt-2"),
                        dbc.Badge(
                             "Pages unknown" if np.isnan(book['num_pages']) else f"{int(book['num_pages'])} pages" , 
                            color="secondary", className="me-2 mt-2"),
                    ], className="mb-2"),
                    html.P(book['genres'], className="small text-muted mb-4"),
                    html.P(
                        book['description'][:250] + "..." if len(str(book['description'])) > 250 else book['description'],
                        
                        className="card-text"
                    ),
                    dbc.Button(
                        "Details",
                        id={"type": "details-button", "index": book.get('work_id', '')},
                        color="secondary",
                        outline=False,
                        size="sm",
                        className="mt-2"
                    )
                ]) 
            ], className="col-12 col-md-10")
        ], className="g-0")
    ], className="mb-3")
    
    return card

# -----------------------
# Callback for theme switching
# -----------------------
@app.callback(
    Output("main-container", "className"),
    Input("theme-switch", "value")
)
def switch_theme(dark_mode):
    if dark_mode:
        return "bg-dark text-white"
    return ""

# -----------------------
# Callback for loved books
# -----------------------
@app.callback(
    Output("loved-books-store", "data"),
    Input({"type": "love-button", "index": dash.ALL}, "n_clicks"),
    State("loved-books-store", "data"),
    prevent_initial_call=True
)
def update_loved_books(n_clicks_list, loved_books):
    if not any(n_clicks_list):
        return loved_books
    
    # Get which button was clicked
    ctx = callback_context
    if not ctx.triggered:
        return loved_books
    
    # Extract button info
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    import json
    button_dict = json.loads(button_id)
    book_id = button_dict["index"]
    
    # Convert loved_books to a list of IDs if needed
    loved_book_ids = []
    for book in loved_books:
        if isinstance(book, dict):
            loved_book_ids.append(book['id'])
        else:
            loved_book_ids.append(book)
    
    # Toggle love status
    if book_id in loved_book_ids:
        # Remove the book
        loved_books = [b for b in loved_books if (isinstance(b, dict) and b['id'] != book_id) or (isinstance(b, str) and b != book_id)]
    else:
        # Add the book (we'll store full data in the export callback)
        loved_books.append(book_id)
    
    return loved_books


# -----------------------
# Callback for search
# -----------------------
@app.callback(
    Output("results-container", "children"),
    [Input("search-button", "n_clicks"),
     Input("query-input", "value"),
     Input("genre-filter", "value"),
    Input("loved-books-store", "data")]
)
def recommend_books(n_clicks, query, selected_genres, loved_books):
    if query:
        query = query.lower().strip()
    
    # 1. Filter by genre
    if selected_genres != None and selected_genres != "All":
        filtered_df = books_df[books_df["genres_lower"].str.contains(selected_genres.lower(), na=False)]
    else:
        filtered_df = books_df
    
    # Check if query matches (part of) any title names
    if query:
        title_matches = filtered_df[filtered_df["original_title_lower"].str.contains(query, na=False)]
    else:
        title_matches = pd.DataFrame()

    if not title_matches.empty:
        # Return title results sorted by publication year (newest first)
        results = title_matches.sort_values("original_publication_year", ascending=False).head(20)[[
            "work_id", "original_title", "author", "genres", "description", "original_publication_year", "avg_rating", "image_url", "num_pages"
        ]].to_dict("records")
    else:
        # If no title match, check if query matches (part of) any author names
        if query:
            author_matches = filtered_df[filtered_df["author_lower"].str.contains(query, na=False)]
        else:
            author_matches = pd.DataFrame()

        if not author_matches.empty:
            # Return author results sorted by publication year (newest first)
            results = author_matches.sort_values("original_publication_year", ascending=False).head(20)[[
                "work_id", "original_title", "author", "genres", "description", "original_publication_year", "avg_rating", "image_url", "num_pages"
            ]].to_dict("records")
        else:
            if query:
                # If no title or author match, fall back to TF-IDF similarity
                filtered_indices = filtered_df.index
                if filtered_indices.empty:
                    return dbc.Alert("No books found matching your criteria.", color="warning")
                
                # Ensure filtered_indices is a NumPy array for correct positional indexing
                filtered_indices_array = filtered_indices.to_numpy()
                
                query_vec = vectorizer.transform([query])
                
                # Calculate similarity only on filtered rows
                similarity_scores = linear_kernel(query_vec, tfidf_matrix[filtered_indices_array]).flatten()
                
                # Get top matches (local to filtered set)
                top_indices_local = similarity_scores.argsort()[-20:][::-1]
                
                # Map back to global DataFrame index
                top_indices_global = filtered_indices_array[top_indices_local]

                # Return TF-IDF results in order of similarity (highest first)
                results = filtered_df.loc[top_indices_global][[
                    "work_id", "original_title", "author", "genres", "description", "original_publication_year", "avg_rating", "image_url", "num_pages"
                ]].to_dict("records")
            else:
                # No query, just show genre-filtered results
                results = filtered_df.sort_values("original_publication_year", ascending=False).head(20)[[
                    "work_id", "original_title", "author", "genres", "description", "original_publication_year", "avg_rating", "image_url", "num_pages"
                ]].to_dict("records")
    
    # Create cards for each book
    if results:
        return [create_book_card(book, loved_books) for book in results]
    else:
        return dbc.Alert("No books found matching your search.", color="warning")

@app.callback(
    Output("download-loves", "data"),
    Input("export-loves-btn", "n_clicks"),
    State("loved-books-store", "data"),
    prevent_initial_call=True
)
def export_loved_books(n_clicks, loved_books):
    if not loved_books:
        return None
    
    # Convert to DataFrame
    loved_data = []
    for book in loved_books:
        if isinstance(book, dict):
            loved_data.append(book)
        else:
            # Handle case where only ID is stored
            parts = book.split('_', 1)  # Split only on first underscore
            loved_data.append({
                'title': parts[0] if len(parts) > 0 else '',
                'author': parts[1] if len(parts) > 1 else '',
                'loved_date': pd.Timestamp.now().strftime('%Y-%m-%d')
            })
    
    if not loved_data:
        return None
    
    df = pd.DataFrame(loved_data)
    return dcc.send_data_frame(df.to_csv, "loved_books.csv", index=False)

# -----------------------
# Client-side callback for modal with details book
# -----------------------
@app.callback(
    [Output("book-details-modal", "is_open"),
     Output("modal-title", "children"),
     Output("modal-body", "children")],
    [Input({"type": "details-button", "index": dash.ALL}, "n_clicks"),
     Input("close-modal", "n_clicks")],
    [State("book-details-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_modal(details_clicks, close_clicks, is_open):
    ctx = callback_context
    
    if not ctx.triggered:
        return False, "", ""
    
    trigger_id = ctx.triggered[0]["prop_id"]
    
    # If close button clicked
    if "close-modal" in trigger_id:
        return False, "", ""
    
    # If details button clicked
    if "details-button" in trigger_id:
        # Check if any button was actually clicked (n_clicks > 0)
        if not any(details_clicks) or all(click is None or click == 0 for click in details_clicks):
            return False, "", ""
        
        # Get which button was clicked
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        import json
        button_dict = json.loads(button_id)
        work_id = button_dict["index"]
        
        
        # Get book details
        book = books_df[books_df['work_id'] == work_id].iloc[0]
        
        # Get reviews for this book
        book_reviews = reviews_df[reviews_df['work_id'] == work_id].copy()
        
        # Convert date_added to datetime and sort by most recent
        book_reviews['date_added'] = pd.to_datetime(book_reviews['date_added'], errors='coerce')
        book_reviews = book_reviews.sort_values('date_added', ascending=False).head(10)
        
        # Create modal content
        modal_title = html.H2(f"{book['original_title']} by {book['author']}")
        
        # Create accordion items for reviews
        accordion_items = []
        for idx, review in book_reviews.iterrows():
            review_date = review['date_added'].strftime('%Y-%m-%d') if pd.notna(review['date_added']) else 'Unknown date'
            review_rating = f"★ {int(review['rating'])}" if pd.notna(review['rating']) else "No rating"
            
            accordion_items.append(
                dbc.AccordionItem(
                    [
                        html.P(review['review_text'])
                    ],
                    title=f"{review_date} - {review_rating}",
                )
            )
        
        modal_body = [
            html.H3("Full description", className="mb-3"),
            html.P(book['description'], className="mb-4"),
            html.Hr(),
            html.H4(f"Recent Reviews ({len(accordion_items)})", className="mb-3"),
            dbc.Accordion(
                accordion_items,
                start_collapsed=True  # All items start collapsed
            ) if accordion_items else html.P("No reviews available for this book.", className="text-muted")
        ]
        
        return True, modal_title, modal_body
    
    return False, "", ""



# -----------------------
# Handle click on author and show all books from the author
# UPDATED: Also handles clear button
# -----------------------
@app.callback(
    Output("query-input", "value"),
    [Input({"type": "author-link", "index": dash.ALL}, "n_clicks"),
     Input("clear-button", "n_clicks")],
    prevent_initial_call=True
)
def update_query(author_clicks_list, clear_clicks):
    ctx = callback_context
    if not ctx.triggered:
        return dash.no_update
    
    trigger_id = ctx.triggered[0]["prop_id"]
    
    # If clear button was clicked
    if "clear-button" in trigger_id:
        return ""
    
    # If author link was clicked
    if "author-link" in trigger_id and any(author_clicks_list):
        # Extract author name from the clicked button
        button_id = ctx.triggered[0]["prop_id"].split(".")[0]
        import json
        button_dict = json.loads(button_id)
        author_name = button_dict["index"]
        return author_name
    
    return dash.no_update



@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-button", "n_clicks"),
     Input("close-help-modal", "n_clicks")],
    [State("help-modal", "is_open")],
    prevent_initial_call=True
)
def toggle_help_modal(help_clicks, close_clicks, is_open):
    ctx = callback_context
    
    if not ctx.triggered:
        return False
    
    trigger_id = ctx.triggered[0]["prop_id"]
    
    # Toggle the modal
    if "help-button" in trigger_id or "close-help-modal" in trigger_id:
        return not is_open
    
    return is_open

# -----------------------
# Client-side callback for theme
# -----------------------
app.clientside_callback(
    """
    function(dark_mode) {
        if (dark_mode) {
            document.body.setAttribute('data-bs-theme', 'dark');
        } else {
            document.body.setAttribute('data-bs-theme', 'light');
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output("theme-store", "data"),
    Input("theme-switch", "value")
)

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    app.run(debug=False)