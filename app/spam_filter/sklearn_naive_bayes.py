def train_spam_filter():
    # Read and concatenate email_data_0.csv and email_data_1.csv

    # Use TfidfVectorizer for fit and transform 'email_body'

    # covert the sparse matrix row to a dense array

    # split dataset for 70% train and 30% test

    # USe 'MultinomialNB' for 'fit' and 'predict'
    labels = ['ham', 'spam']

    # Use 'precision_recall_fscore_support' for get and print (precision, recall, fscore, support)

    # Print 'classification_report'

    return tfidf_vect, naive_byes_clf


def predict_new(tfidf_vect, clf, new_emails):
    labels = ['ham', 'spam']
    # Use tfidf_vect for the text transformation
    new_text_tfidf = tfidf_vect.transform(new_emails)

    print("New (income) emails amount: %s train data amount: %s" % new_text_tfidf.shape)
    # Predict probability with pre-trained MultinomialNB
    predicted_p = clf.predict_proba(new_text_tfidf)
    predicted = clf.predict(new_text_tfidf)

    # Print probability results
    for idx, doc in enumerate(new_emails):
        print("Email body: %s" % doc)
        for j, label in enumerate(labels):
            print('%s: %.3f' % (labels[j], predicted_p[idx][j]))
        print('The email is => %s\n' % (labels[predicted[idx]]))


if __name__ == '__main__':
    """
        generate tfidf matrix and a NB model
    """
    tfidf_vect, naive_byes_clf = train_spam_filter()

    """
        input : list of email text
    """
    new_emails = [
        'awordthatdoesnotexist письмо',
        "Text82228>> Get more ringtones, logos and games from www.txt82228.com. Questions: info@txt82228.co.uk",  # spam
        "Just looked it up and addie goes back Monday, sucks to be her",  # ham
        "You have won a guaranteed price`",
        "Dear student you are almost done ML workshop"
    ]
    """
        output: 
    """
    predict_new(tfidf_vect, naive_byes_clf, new_emails)
