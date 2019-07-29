jane_tidy <- austen_books() %>%
  filter(book == "Emma") %>%
  mutate(linenumber = row_number()) %>% 
  mutate(chapter = str_detect(text, "CHAPTER") %>% cumsum())

jane_sent_df <- jane_tidy %>%
  get_sentences() %>%
  sentiment_by(by = colnames(jane_tidy), polarity_dt = lexicon::hash_sentiment_jockers_rinker)

jane_tokens <- space_tokenizer(
  jane_tidy$text %>% tolower() %>% removePunctuation()
  )

jane_it <- itoken(jane_tokens, progressbar = FALSE)
jane_vocab <- create_vocabulary(jane_it)
jane_vocab <- prune_vocabulary(jane_vocab, term_count_min = 3L)
jane_vectorizer <- vocab_vectorizer(jane_vocab)
jane_tcm <- create_tcm(jane_it, jane_vectorizer, skip_grams_window = 5L)

jane_glove <- GloVe$new(word_vectors_size = 100, vocabulary = jane_vocab, x_max = 5)
jane_glove$fit_transform(jane_tcm, n_iter = 20)
jane_word_vectors <- jane_glove$components

jane_emma <- jane_word_vectors[, "emma", drop = F] 

jane_cos_sim <- sim2(
  x = t(jane_word_vectors), 
  y = t(jane_emma), 
  method = "cosine", 
  norm = "l2"
  )
head(sort(jane_cos_sim[,1], decreasing = TRUE), 10)

# create vector of words to keep, before applying tsne (i.e. remove stop words)
jane_keep_words <- setdiff(colnames(jane_word_vectors), stopwords())

# keep words in vector
jane_word_vec <- jane_word_vectors[, jane_keep_words]

# prepare data frame to train
jane_train_df <- data.frame(t(jane_word_vec)) %>%
  rownames_to_column("word")

# train tsne for visualization
jane_tsne <- Rtsne(jane_train_df[,-1], dims = 2, perplexity = 50, verbose=TRUE, max_iter = 500)

# create plot
jane_colors = rainbow(length(unique(jane_train_df$word)))
names(jane_colors) = unique(jane_train_df$word)

jane_plot_df <- data.frame(jane_tsne$Y) %>%
  mutate(
    word = jane_train_df$word,
    col = jane_colors[jane_train_df$word]
  ) %>%
  left_join(jane_vocab, by = c("word" = "term")) %>%
  filter(doc_count >= 20)

ggplot(jane_plot_df, aes(X1, X2)) +
  geom_text(aes(X1, X2, label = word, color = col), size = 3) +
  xlab("") + ylab("") +
  theme(legend.position = "none") 

# calculate word-level sentiment
jane_word_sent <- jane_tidy %>%
  left_join(jane_sent_df, by = c("text", "book", "linenumber", "chapter")) %>%
  unnest_tokens(word, text) %>%
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
  jane_pd_sent <- jane_plot_df %>%
    left_join(jane_word_sent, by = "word") %>%
    filter(count >= 5)
  
  # create plot
  ggplot(jane_pd_sent, aes(X1, X2)) +
    geom_point(aes(X1, X2, size = count, alpha = .1, color = avg_sentiment)) +
    geom_text(aes(X1, X2, label = word), size = 2) +
    scale_colour_gradient2(low = muted("red"), mid = "white",
                           high = muted("blue"), midpoint = 0) +
    scale_size(range = c(5, 20)) +
    xlab("") + ylab("") +
    ggtitle("2-dimensional t-SNE Mapping of Word Vectors") +
    guides(color = guide_legend(title="Avg. Sentiment"), size = guide_legend(title = "Frequency"), alpha = NULL) +
    scale_alpha(range = c(1, 1), guide = "none")
  
  ggplot(jane_pd_sent, aes(X1, X2)) +
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
