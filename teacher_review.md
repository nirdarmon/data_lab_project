# Supervisor review of the proposal

Feedback from the supervisor on `proposal.ipynb`. The proposal is **approved**. The
next phase (the final report, `final.ipynb`) addresses only the points below.

## Action items for the final report

1. **Loss for extreme imbalance** (~98.6% background). Justify the loss choice: BCE +
   soft Dice (current), and consider Tversky / Focal Tversky to weight recall so thin
   cracks are not missed. Answer his closing question explicitly in the notebook.
2. **Patch sampling ratio.** Fix an explicit train ratio of crack to background patches
   (he suggests 60% crack / 40% clean background), balancing crack exposure against
   false positives on rough concrete.
3. **Pretrained encoder.** Compare the plain U-Net with a U-Net on a pretrained backbone
   (e.g. ResNet34 or EfficientNet).
4. **Post-processing.** Morphological closing, removing small connected components, or
   skeletonization, and measure the effect on Dice / IoU.

## Original text (Hebrew, verbatim)

הי ניר, עמית ואלה

הצעת הפרויקט שלכם בנושא זיהוי ומיקום סדקים בבטון (Concrete Crack Detection and Localization Using Deep Learning) היא הצעה מצוינת, בעלת בשלות מתודולוגית והנדסית, ומאושרת להמשך עבודה.

החיבור בין הבעיה ההנדסית של ניטור מבני (Structural Health Monitoring) לבין ניסוח המשימה כסגמנטציה סמנטית ברמת הפיקסל מנומק היטב, וההקפדה על איכות הנתונים כבר בשלב זה מרשימה במיוחד.

### נקודות חוזק

* ניסוח נכון ומעשי של הבעיה: ההחלטה להתמקד בסגמנטציה סמנטית (Binary Mask) ולא בסיווג בינארי של התמונה כולה מוצדקת ומדויקת. במבני בטון ישנים כמעט תמיד קיים פגם כלשהו, ורק חילוץ מסכה מאפשר גזירת מדדים הנדסיים קריטיים כמו אורך, רוחב מקסימלי, שטח וקצב גדילה לאורך זמן.
* בקרת איכות נתונים (Data Hygiene) ברמה הגבוהה ביותר:
  * איתור בעיית ה-EXIF Orientation: זיהוי העובדה שתגיות האוריינטציה של קובצי ה-JPEG הובילו להיפוך ולחוסר התאמה מול המסכות המקוריות היא נקודת זכות גדולה. יתרה מזו, הפיתוח של מדד התאמה פנימי (בדיקת פער הכהות בפיקסלי הסדק לעומת הרקע) מוכיח חשיבה מדעית יסודית שאינה מסתפקת רק בהשוואת ממדי התמונה.
  * בינריזציה של מסכות: זיהוי רעשי הדחיסה בקובצי המסכה וקביעת סף חיתוך של 127 מבטיחים מטרות אימון (ground truth) נקיות וחד-משמעיות.
* מתודולוגיית חלוקה קפדנית ומניעת Data Leakage:
  * ביצוע החלוקה ברמת תמונת המקור ולא ברמת הפאצ'ים מונע דליפת מידע בין תתי-אזורים של אותה תמונה.
  * השימוש ב-Perceptual Hashing (pHash) לאיתור כפילויות כמעט-זהות (near-duplicates) ואיגודן לקבוצות שלמות לפני החלוקה (Group Split), יחד עם ריבוד (Stratification) לפי אחוז שטח הסדק, מציג סטנדרט עבודה מחקרי גבוה.
* בחירת מדדי הצלחה רלוונטיים: ההתמקדות ב-Dice/F1 ו-IoU במקום ב-Accuracy מותאמת ישירות לאופי הנתונים, שבהם הסדק מהווה בממוצע רק כ-1.36% מסך הפיקסלים.

### דגשים והמלצות לקראת שלב המידול והדוח הסופי

* פונקציות הפסד לחוסר איזון קיצוני: תחת יחס של כ-98.6% רקע, אימון באמצעות Cross-Entropy סטנדרטי עלול להתכנס לחיזוי רקע בלבד. יש לבסס את האימון על שילובים מותאמים כגון BCE + Soft Dice Loss או Tversky / Focal Tversky Loss, המאפשרים לתת משקל יתר ל-Recall כדי למנוע החמצת סדקים דקים.
* אסטרטגיית דגימת פאצ'ים (Patch Sampling): בחיתוך אחיד של פאצ'ים (256x256) על פני כל שטח התמונה, רוב הפאצ'ים יהיו ריקים לחלוטין מסדקים. מומלץ להגדיר יחס מאוזן מראש באימון (למשל 60% פאצ'ים המכילים סדק ו-40% פאצ'י רקע נקיים), כדי לאזן בין חשיפת המודל לדפוסי סדקים לבין אימון המודל לא להקפיץ False Positives על פני משטחי בטון מחוספסים.
* ארכיטקטורת U-Net ו-Backbone מאומן מראש: מעבר ל-U-Net בסיסי, כדאי לבחון שילוב של Encoder מאומן מראש (Pretrained Backbone, כדוגמת ResNet34 או EfficientNet). משקלים מאומנים מסייעים רבות בחילוץ מאפייני שוליים וטקסטורה כבר בשכבות המוקדמות.
* עיבוד שלאחר מידול (Post-Processing): סדקים מתאפיינים ברציפות טופולוגית. שימוש בפעולות מורפולוגיות קלאסיות (כגון סגירה מורפולוגית, הסרת רכיבים קשירים קטנים או שלד סדק) עשוי לנקות רעשים נקודתיים ולשפר את מדדי ה-IoU וה-Dice.

באיזו פונקציית הפסד משולבת (כגון Combo Loss של BCE ו-Dice, או Tversky Loss) אתם מתכננים להשתמש כדי להתמודד עם חוסר האיזון הקיצוני שבו 98.6% מהפיקסלים הם רקע?

זאב
