# Error Troubleshooting

- **ValueError: could not convert string to float: 'BMW'**  
  Caused by categorical strings in numerical models. Solution: encode text fields before passing to model.

- **TypeError: generate_full_explanation() takes 2 positional arguments but 3 were given**  
  Caused by mismatched function signature. Adjust wrapper to support optional extra_filters.

- **AttributeError: st.session_state has no attribute 'memory'**  
  Streamlit requires explicit initialization of session state variables.

These notes help the chat agent guide users when errors appear.
