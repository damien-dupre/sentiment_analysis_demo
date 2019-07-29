# libraries --------------------------------------------------------------------
library(tidyverse)
library(data.table)
library(sentimentr)
library(text2vec)
library(tm)
library(ggrepel)
library(Rtsne)
library(tidytext)
library(scales)
# grab reviews
reviews_all <- read.csv("https://raw.githubusercontent.com/rjsaito/Just-R-Things/master/NLP/sample_reviews_venom.csv", stringsAsFactors = F)

# create ID for reviews
review_df <- reviews_all %>%
  mutate(id = row_number())

# get n rows
nrow(lexicon::hash_sentiment_jockers_rinker)

# example lexicon
head(lexicon::hash_sentiment_jockers_rinker)

# words to replace
replace_in_lexicon <- tribble(
  ~x, ~y,
  "marvel", 0,  # original score: .75
  "venom", 0,   # original score: -.75
  "alien", 0,   # original score: -.6
  "bad@$$", .4, # not in dictionary
  "carnage", 0, # original score: 0.75
  "avenger", 0, # original score: .25
  "riot", 0     # original score: -.5
)

# create a new lexicon with modified sentiment
venom_lexicon <- lexicon::hash_sentiment_jockers_rinker %>%
  filter(!x %in% replace_in_lexicon$x) %>%
  bind_rows(replace_in_lexicon) %>%
  setDT() %>%
  setkey("x")

# get sentence-level sentiment
sent_df <- review_df %>%
  get_sentences() %>%
  sentiment_by(by = c('id', 'author', 'date', 'stars', 'review_format'), polarity_dt = venom_lexicon)


# stars vs sentiment
ggplot(sent_df, aes(x = stars, y = ave_sentiment, color = factor(stars), group = stars)) +
  geom_boxplot() +
  geom_hline(yintercept=0, linetype="dashed", color = "red") +
  geom_text(aes(5.2, -0.05, label = "Neutral Sentiment", vjust = 0), size = 3, color = "red") +
  guides(color = guide_legend(title="Star Rating")) +
  ylab("Avg. Sentiment") +
  xlab("Review Star Rating") +
  ggtitle("Sentiment of Venom Amazon Reviews, by Star Rating") 

# sentiment over time
sent_ts <- sent_df %>%
  mutate(
    date = as.Date(date, format = "%d-%B-%y"),
    dir = sign(ave_sentiment)
  ) %>%
  group_by(date) %>%
  summarise(
    avg_sent = mean(ave_sentiment)
  ) %>%
  ungroup()

# plot
ggplot(sent_ts, aes(x = date, y = avg_sent)) +
  geom_smooth(method="loess", size=1, se=T, span = .6) +
  geom_vline(xintercept=as.Date("2018-10-05"), linetype="dashed", color = "black") +
  geom_text(aes(as.Date("2018-10-05") + 1, .4, label = "Theatrical Release", hjust = "left"), size = 3, color = "black") +
  geom_vline(xintercept=as.Date("2018-12-11"), linetype="dashed", color = "black") +
  geom_text(aes(as.Date("2018-12-11") + 1, .4, label = "Digital Release", hjust = "left"), size = 3, color = "black") +
  geom_vline(xintercept=as.Date("2018-12-18"), linetype="dashed", color = "black") +
  geom_text(aes(as.Date("2018-12-18") + 1, .37, label = "DVD / Blu-Ray Release", hjust = "left"), size = 3, color = "black") +
  geom_hline(yintercept=0, linetype="dashed", color = "red") +
  geom_text(aes(max(sent_ts$date) - 5, -0.02, label = "Neutral Sentiment", vjust = 0), size = 3, color = "red") +
  ylab("Avg. Sentiment") +
  xlab("Review Date") +
  ggtitle("Sentiment of Venom Amazon Reviews, Over Time (2018 - 2019)")

# create lists of reviews split into individual words (iterator over tokens)
tokens <- space_tokenizer(reviews_all$comments %>%
                            tolower() %>%
                            removePunctuation())

# Create vocabulary. Terms will be unigrams (simple words).
it <- itoken(tokens, progressbar = FALSE)
vocab <- create_vocabulary(it)

# prune words that appear less than 3 times
vocab <- prune_vocabulary(vocab, term_count_min = 3L)

# Use our filtered vocabulary
vectorizer <- vocab_vectorizer(vocab)

# use skip gram window of 5 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window = 5L)

# fit the model. (It can take several minutes to fit!)
glove = GloVe$new(word_vectors_size = 100, vocabulary = vocab, x_max = 5)
glove$fit_transform(tcm, n_iter = 20)

# obtain word vector
word_vectors = glove$components

brock <- word_vectors[, "brock", drop = F] 

cos_sim = sim2(x = t(word_vectors), y = t(brock), method = "cosine", norm = "l2")
head(sort(cos_sim[,1], decreasing = TRUE), 10)

# create vector of words to keep, before applying tsne (i.e. remove stop words)
keep_words <- setdiff(colnames(word_vectors), stopwords())

# keep words in vector
word_vec <- word_vectors[, keep_words]

# prepare data frame to train
train_df <- data.frame(t(word_vec)) %>%
  rownames_to_column("word")

# train tsne for visualization
tsne <- Rtsne(train_df[,-1], dims = 2, perplexity = 50, verbose=TRUE, max_iter = 500)

# create plot
colors = rainbow(length(unique(train_df$word)))
names(colors) = unique(train_df$word)

plot_df <- data.frame(tsne$Y) %>%
  mutate(
    word = train_df$word,
    col = colors[train_df$word]
  ) %>%
  left_join(vocab, by = c("word" = "term")) %>%
  filter(doc_count >= 20)

ggplot(plot_df, aes(X1, X2)) +
  geom_text(aes(X1, X2, label = word, color = col), size = 3) +
  xlab("") + ylab("") +
  theme(legend.position = "none") 

# calculate word-level sentiment
word_sent <- review_df %>%
  left_join(sent_df, by = "id") %>%
  select(id, comments, ave_sentiment) %>%
  unnest_tokens(word, comments) %>%
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
pd_sent <- plot_df %>%
  left_join(word_sent, by = "word") %>%
  drop_na() %>%
  filter(count >= 5)

# create plot
ggplot(pd_sent, aes(X1, X2)) +
  geom_point(aes(X1, X2, size = count, alpha = .1, color = avg_sentiment)) +
  geom_text(aes(X1, X2, label = word), size = 2) +
  scale_colour_gradient2(low = muted("red"), mid = "white",
                         high = muted("blue"), midpoint = 0) +
  scale_size(range = c(5, 20)) +
  xlab("") + ylab("") +
  ggtitle("2-dimensional t-SNE Mapping of Word Vectors") +
  guides(color = guide_legend(title="Avg. Sentiment"), size = guide_legend(title = "Frequency"), alpha = NULL) +
  scale_alpha(range = c(1, 1), guide = "none")
