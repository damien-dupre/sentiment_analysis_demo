library(xml2)
library(rvest)
library(stringr)
library(tibble)
library(tidyr)
library(dplyr)


url <- "https://www.imdb.com/title/tt0092610/reviews/_ajax"

data_key <- url %>% 
  xml2::read_html() %>%
  rvest::html_nodes(".load-more-data") %>% 
  xml2::xml_attr("data-key")

review_all <- data.frame()

while(length(data_key)==1){
  review_content <- url %>% 
    xml2::read_html() %>%
    rvest::html_nodes(".review-container") %>%
    rvest::html_text() %>% 
    stringr::str_replace_all("  ","") %>% 
    stringr::str_replace_all("\n","") %>% 
    stringr::str_replace_all("found this helpful.Was this review helpful?Sign in to vote.Permalink", "")
  
  review_data <- url %>% 
    xml2::read_html() %>%
    rvest::html_nodes(".display-name-date") %>%
    rvest::html_text() %>% 
    stringr::str_trim() %>% 
    tibble::enframe(name = "review") %>% 
    tidyr::separate(value, c("name", "month", "year"), sep = " ") %>% 
    dplyr::mutate(content = review_content)
  
  review_all <- review_all %>% 
    dplyr::bind_rows(review_data)
  
  data_key <- url %>% 
    xml2::read_html() %>%
    rvest::html_nodes(".load-more-data") %>% 
    xml2::xml_attr("data-key")
  
  url <- paste0("https://www.imdb.com/title/tt0092610/reviews/_ajax?ref_=undefined&paginationKey=", data_key)
}

# data analysis ----------------------------------------------------------------
df <- review_all %>% 
  dplyr::mutate(year = as.numeric(year)) %>% 
  na.omit() %>% 
  dplyr::mutate(timestamp = zoo::as.yearmon(paste(year, month), "%Y %B")) %>% 
  dplyr::select(-review) %>% 
  tibble::rowid_to_column("review")
library(tidytext)
test_df <- df %>% 
  unnest_tokens(word, content)

test_df2 <- test_df %>% 
  anti_join(stop_words)

afinn_sentiments <- get_sentiments("afinn")

test_df3 <- test_df2 %>% 
  inner_join(afinn_sentiments) %>% 
  group_by(name, timestamp) %>% 
  summarise(avg_score = mean(score))

test_df3 %>%
  ggplot(aes(timestamp, avg_score)) + 
  geom_point() + 
  geom_smooth()

library(tidyverse)
library(data.table)
library(sentimentr)
library(text2vec)
library(tm)
library(ggrepel)
library(Rtsne)
library(tidytext)
library(scales)

review_sent_df <- review_all %>%
  sentimentr::get_sentences() %>%
  sentimentr::sentiment_by(by = colnames(review_all), polarity_dt = lexicon::hash_sentiment_jockers_rinker) %>% 
  dplyr::filter(content != "Sign in to vote.Permalink") %>% 
  dplyr::filter(content != "Was this review helpful?") %>% 
  dplyr::filter(content != str_detect(content, "found this helpful"))

review_tokens <- text2vec::space_tokenizer(
  review_all$content %>% tolower() %>% tm::removePunctuation()
)

review_it <- itoken(review_tokens, progressbar = FALSE)
review_vocab <- create_vocabulary(review_it)
review_vocab <- prune_vocabulary(review_vocab, term_count_min = 3L)
review_vectorizer <- vocab_vectorizer(review_vocab)
review_tcm <- create_tcm(review_it, review_vectorizer, skip_grams_window = 5L)

review_glove <- GloVe$new(word_vectors_size = 100, vocabulary = review_vocab, x_max = 5)
review_glove$fit_transform(review_tcm, n_iter = 20)
review_word_vectors <-  review_glove$components

review_derek <- review_word_vectors[, "derek", drop = F] 

review_cos_sim <- sim2(
  x = t(review_word_vectors), 
  y = t(review_derek), 
  method = "cosine", 
  norm = "l2"
)
head(sort(review_cos_sim[,1], decreasing = TRUE), 10)

# create vector of words to keep, before applying tsne (i.e. remove stop words)
review_keep_words <- setdiff(colnames(review_word_vectors), stopwords())

# keep words in vector
review_word_vec <- review_word_vectors[, review_keep_words]

# prepare data frame to train
review_train_df <- data.frame(t(review_word_vec)) %>%
  rownames_to_column("word")

# train tsne for visualization
review_tsne <- Rtsne(review_train_df[,-1], dims = 2, perplexity = 50, verbose=TRUE, max_iter = 500)

# create plot
review_colors = rainbow(length(unique(review_train_df$word)))
names(review_colors) = unique(review_train_df$word)

review_plot_df <- data.frame(review_tsne$Y) %>%
  mutate(
    word = review_train_df$word,
    col = review_colors[review_train_df$word]
  ) %>%
  left_join(review_vocab, by = c("word" = "term")) %>%
  filter(doc_count >= 20)

ggplot(review_plot_df, aes(X1, X2)) +
  geom_text(aes(X1, X2, label = word, color = col), size = 3) +
  xlab("") + ylab("") +
  theme(legend.position = "none") 

# calculate word-level sentiment
review_word_sent <- review_all %>%
  left_join(review_sent_df) %>%
  unnest_tokens(word, content) %>%
  group_by(word) %>%
  summarise(
    count = n(),
    avg_sentiment = mean(ave_sentiment),
    sum_sentiment = sum(ave_sentiment),
    sd_sentiment = sd(ave_sentiment)
  ) %>%
  # rtemove stop words
  anti_join(stop_words, by = "word")



# filter to words that appear at least 5 times
review_pd_sent <- review_plot_df %>%
  left_join(review_word_sent, by = "word") %>%
  filter(count >= 5)

# create plot
ggplot(review_pd_sent, aes(X1, X2)) +
  geom_point(aes(X1, X2, size = count, alpha = .1, color = avg_sentiment)) +
  geom_text(aes(X1, X2, label = word), size = 2) +
  scale_colour_gradient2(low = muted("red"), mid = "white",
                         high = muted("blue"), midpoint = 0) +
  scale_size(range = c(5, 20)) +
  xlab("") + ylab("") +
  ggtitle("2-dimensional t-SNE Mapping of Word Vectors") +
  guides(color = guide_legend(title="Avg. Sentiment"), size = guide_legend(title = "Frequency"), alpha = NULL) +
  scale_alpha(range = c(1, 1), guide = "none")

ggplot(review_pd_sent, aes(X1, X2)) +
  geom_point(aes(X1, X2, size = count, alpha = .1, color = col)) +
  geom_text(aes(X1, X2, label = word), size = 2) +
  # scale_colour_gradient2(low = muted("red"), mid = "white",
  #                        high = muted("blue"), midpoint = 0) +
  scale_size(range = c(5, 20)) +
  xlab("") + ylab("") +
  ggtitle("2-dimensional t-SNE Mapping of Word Vectors") +
  guides(color = guide_legend(title="Avg. Sentiment"), size = guide_legend(title = "Frequency"), alpha = NULL) +
  scale_alpha(range = c(1, 1), guide = "none") +
  theme(legend.position = "none")
