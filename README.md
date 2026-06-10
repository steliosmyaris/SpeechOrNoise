The project was to train an ai model on data from various sources.
I haven't uploaded the data because they were huge. 
The repository includes the results of the code but you can have them yourself by running 
train.py and then evaluate.py in this order. 
Requirements:
pip install librosa scikit-learn scipy pandas numpy joblib

Θέμα: Καλείστε να υλοποιήσετε ένα σύστημα που προχωρά στην κατάτμηση μιας πρότασης σε
τμήματα σήματος ομιλίας (foreground) και σήματος υποβάθρου (background), χρησιμοποιώντας
υποχρεωτικά έναν ταξινομητή background vs foreground (ανά frame) και εφαρμόζοντας στη
συνέχεια μετα-επεξεργασία της ακολουθίας αποφάσεων του ταξινομητή. Τελικώς, δοθείσης μιας
ηχογράφησης, το σύστημα επιστρέφει τα χρονικά όρια των τμημάτων σήματος ομιλίας και των
τμημάτων σήματος υποβάθρου (σε δευτερόλεπτα). To format των αποτελεσμάτων είναι ένα αρχείο
csv, της μορφής:
Audiofile, start, end, class
File1, 0, 2.2, background
File1, 2.2, 4, foreground
File1, 4,10, background
…
Θα πρέπει να υλοποιήσετε και να σχολιάσετε τις επιδόσεις των παρακάτω ταξινομητών: k-NN και
MLP δύο στρωμάτων (αποφασίστε το πλήθος νευρώνων ανά στρώμα).
Για την εκπαίδευση, παρέχεται στο link
https://drive.google.com/drive/folders/1A-_ybw6sVtPYjrzkOa1rACvvCyqPqhWn?usp=share_link
ένας κατάλογος αρχείων train, με υποκαταλόγους ομιλίας (speech) και υποβάθρου (noise). Μπορείτε
να χρησιμοποιήσετε ένα υποσύνολο των αρχείων αυτών. Για την τελική δοκιμή παρέχεται ένας
κατάλογος test με ένα μεικτό αρχείο, καθώς και τα σχετικά transcriptions σε json format
(υποκατάλογος transcriptions). Από τα json αρχεία, θα χρειαστεί να διατηρήσετε μόνο τα δεδομένα
που αφορούν στο test αρχείο που δίνεται και να αγνοήσετε την υπόλοιπη πληροφορία. Όλα τα
αρχεία προέρχονται από γνωστά, δημόσια σύνολα δεδομένων (https://www.openslr.org, συλλογές
MUSAN και CHiME) και ίσως χρειαστεί η τμηματική επεξεργασία τους, αναλόγως του τρόπου
ανάλυσης που θα επιλέξετε. Ανεξαρτήτως προσέγγισης, προσπαθήστε να ποσοτικοποιήσετε
κατάλληλα τις επιδόσεις των ταξινομητών.
