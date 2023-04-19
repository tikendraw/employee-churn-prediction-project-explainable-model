#! bin/bash

# installing nltk addons
git clone https://github.com/tikendraw/funcyou.git -q

# setting up streamlit
mkdir -p ~/.streamlit/ 
    echo "\ [server]\n\
    headless = true\n\
    port = $PORT\n\
    enableCORS = false\n\
    \n\" > ~/.streamlit/config.toml
