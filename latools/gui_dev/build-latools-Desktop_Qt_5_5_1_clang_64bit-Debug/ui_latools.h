/********************************************************************************
** Form generated from reading UI file 'latools.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_LATOOLS_H
#define UI_LATOOLS_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QCheckBox>
#include <QtWidgets/QComboBox>
#include <QtWidgets/QFrame>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QListWidget>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QMenu>
#include <QtWidgets/QMenuBar>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStatusBar>
#include <QtWidgets/QTabWidget>
#include <QtWidgets/QToolBar>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_latools
{
public:
    QAction *actionLoad_Data;
    QAction *actionLoad_Data_2;
    QAction *actionDespike;
    QAction *actionIdentify_Background;
    QAction *actionBackground_Correct;
    QAction *actionRatio;
    QAction *actionIdentify_Standards;
    QAction *actionApply_SRMs;
    QWidget *centralWidget;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout;
    QTabWidget *tabWidget;
    QWidget *tab_load;
    QGridLayout *gridLayout_2;
    QGridLayout *gridLayout;
    QPushButton *pushButton;
    QPushButton *pushButton_2;
    QLabel *label_3;
    QLabel *label_4;
    QWidget *tab_bkg;
    QGridLayout *gridLayout_4;
    QGridLayout *gridLayout_3;
    QCheckBox *checkBox_4;
    QComboBox *comboBox;
    QPushButton *pushButton_3;
    QLabel *label_5;
    QWidget *tab_srm;
    QGridLayout *gridLayout_6;
    QGridLayout *gridLayout_5;
    QPushButton *pushButton_5;
    QPushButton *pushButton_4;
    QLabel *label_6;
    QLabel *label_7;
    QWidget *tab_stat;
    QGridLayout *gridLayout_8;
    QGridLayout *gridLayout_7;
    QCheckBox *checkBox_10;
    QCheckBox *checkBox_6;
    QCheckBox *checkBox_8;
    QCheckBox *checkBox_5;
    QHBoxLayout *horizontalLayout_2;
    QCheckBox *checkBox_9;
    QComboBox *comboBox_2;
    QCheckBox *checkBox_7;
    QPushButton *pushButton_6;
    QPushButton *pushButton_7;
    QLabel *label_8;
    QWidget *widget;
    QHBoxLayout *plotLayout;
    QVBoxLayout *optionsPane;
    QLabel *label;
    QCheckBox *checkBox;
    QCheckBox *checkBox_3;
    QCheckBox *checkBox_2;
    QFrame *line;
    QLabel *label_2;
    QListWidget *listWidget;
    QFrame *line_2;
    QSpacerItem *verticalSpacer;
    QWidget *plotPane;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QMenu *menuEdit;
    QMenu *menuProcess;
    QMenu *menuHelp;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *latools)
    {
        if (latools->objectName().isEmpty())
            latools->setObjectName(QStringLiteral("latools"));
        latools->resize(1221, 774);
        actionLoad_Data = new QAction(latools);
        actionLoad_Data->setObjectName(QStringLiteral("actionLoad_Data"));
        actionLoad_Data_2 = new QAction(latools);
        actionLoad_Data_2->setObjectName(QStringLiteral("actionLoad_Data_2"));
        actionDespike = new QAction(latools);
        actionDespike->setObjectName(QStringLiteral("actionDespike"));
        actionIdentify_Background = new QAction(latools);
        actionIdentify_Background->setObjectName(QStringLiteral("actionIdentify_Background"));
        actionBackground_Correct = new QAction(latools);
        actionBackground_Correct->setObjectName(QStringLiteral("actionBackground_Correct"));
        actionRatio = new QAction(latools);
        actionRatio->setObjectName(QStringLiteral("actionRatio"));
        actionIdentify_Standards = new QAction(latools);
        actionIdentify_Standards->setObjectName(QStringLiteral("actionIdentify_Standards"));
        actionApply_SRMs = new QAction(latools);
        actionApply_SRMs->setObjectName(QStringLiteral("actionApply_SRMs"));
        centralWidget = new QWidget(latools);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        centralWidget->setMouseTracking(false);
        verticalLayout = new QVBoxLayout(centralWidget);
        verticalLayout->setSpacing(2);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        verticalLayout->setContentsMargins(2, 2, 2, 2);
        horizontalLayout = new QHBoxLayout();
        horizontalLayout->setSpacing(6);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalLayout->setSizeConstraint(QLayout::SetMaximumSize);
        tabWidget = new QTabWidget(centralWidget);
        tabWidget->setObjectName(QStringLiteral("tabWidget"));
        QSizePolicy sizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(tabWidget->sizePolicy().hasHeightForWidth());
        tabWidget->setSizePolicy(sizePolicy);
        tab_load = new QWidget();
        tab_load->setObjectName(QStringLiteral("tab_load"));
        gridLayout_2 = new QGridLayout(tab_load);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QStringLiteral("gridLayout_2"));
        gridLayout = new QGridLayout();
        gridLayout->setSpacing(6);
        gridLayout->setObjectName(QStringLiteral("gridLayout"));
        pushButton = new QPushButton(tab_load);
        pushButton->setObjectName(QStringLiteral("pushButton"));

        gridLayout->addWidget(pushButton, 0, 0, 1, 1);

        pushButton_2 = new QPushButton(tab_load);
        pushButton_2->setObjectName(QStringLiteral("pushButton_2"));

        gridLayout->addWidget(pushButton_2, 1, 0, 1, 1);

        label_3 = new QLabel(tab_load);
        label_3->setObjectName(QStringLiteral("label_3"));

        gridLayout->addWidget(label_3, 0, 1, 1, 1);

        label_4 = new QLabel(tab_load);
        label_4->setObjectName(QStringLiteral("label_4"));

        gridLayout->addWidget(label_4, 1, 1, 1, 1);

        gridLayout->setColumnStretch(0, 1);
        gridLayout->setColumnStretch(1, 4);

        gridLayout_2->addLayout(gridLayout, 0, 0, 1, 1);

        tabWidget->addTab(tab_load, QString());
        tab_bkg = new QWidget();
        tab_bkg->setObjectName(QStringLiteral("tab_bkg"));
        gridLayout_4 = new QGridLayout(tab_bkg);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QStringLiteral("gridLayout_4"));
        gridLayout_3 = new QGridLayout();
        gridLayout_3->setSpacing(6);
        gridLayout_3->setObjectName(QStringLiteral("gridLayout_3"));
        checkBox_4 = new QCheckBox(tab_bkg);
        checkBox_4->setObjectName(QStringLiteral("checkBox_4"));

        gridLayout_3->addWidget(checkBox_4, 1, 0, 1, 1);

        comboBox = new QComboBox(tab_bkg);
        comboBox->setObjectName(QStringLiteral("comboBox"));

        gridLayout_3->addWidget(comboBox, 0, 1, 1, 1);

        pushButton_3 = new QPushButton(tab_bkg);
        pushButton_3->setObjectName(QStringLiteral("pushButton_3"));

        gridLayout_3->addWidget(pushButton_3, 0, 0, 1, 1);

        label_5 = new QLabel(tab_bkg);
        label_5->setObjectName(QStringLiteral("label_5"));

        gridLayout_3->addWidget(label_5, 0, 2, 1, 1);

        gridLayout_3->setColumnStretch(0, 1);
        gridLayout_3->setColumnStretch(1, 1);
        gridLayout_3->setColumnStretch(2, 3);

        gridLayout_4->addLayout(gridLayout_3, 0, 1, 1, 1);

        tabWidget->addTab(tab_bkg, QString());
        tab_srm = new QWidget();
        tab_srm->setObjectName(QStringLiteral("tab_srm"));
        gridLayout_6 = new QGridLayout(tab_srm);
        gridLayout_6->setSpacing(6);
        gridLayout_6->setContentsMargins(11, 11, 11, 11);
        gridLayout_6->setObjectName(QStringLiteral("gridLayout_6"));
        gridLayout_5 = new QGridLayout();
        gridLayout_5->setSpacing(6);
        gridLayout_5->setObjectName(QStringLiteral("gridLayout_5"));
        pushButton_5 = new QPushButton(tab_srm);
        pushButton_5->setObjectName(QStringLiteral("pushButton_5"));

        gridLayout_5->addWidget(pushButton_5, 1, 0, 1, 1);

        pushButton_4 = new QPushButton(tab_srm);
        pushButton_4->setObjectName(QStringLiteral("pushButton_4"));

        gridLayout_5->addWidget(pushButton_4, 0, 0, 1, 1);

        label_6 = new QLabel(tab_srm);
        label_6->setObjectName(QStringLiteral("label_6"));

        gridLayout_5->addWidget(label_6, 0, 1, 1, 1);

        label_7 = new QLabel(tab_srm);
        label_7->setObjectName(QStringLiteral("label_7"));

        gridLayout_5->addWidget(label_7, 1, 1, 1, 1);

        gridLayout_5->setColumnStretch(0, 1);
        gridLayout_5->setColumnStretch(1, 4);

        gridLayout_6->addLayout(gridLayout_5, 0, 0, 1, 1);

        tabWidget->addTab(tab_srm, QString());
        tab_stat = new QWidget();
        tab_stat->setObjectName(QStringLiteral("tab_stat"));
        gridLayout_8 = new QGridLayout(tab_stat);
        gridLayout_8->setSpacing(6);
        gridLayout_8->setContentsMargins(11, 11, 11, 11);
        gridLayout_8->setObjectName(QStringLiteral("gridLayout_8"));
        gridLayout_7 = new QGridLayout();
        gridLayout_7->setSpacing(6);
        gridLayout_7->setObjectName(QStringLiteral("gridLayout_7"));
        gridLayout_7->setContentsMargins(0, -1, -1, -1);
        checkBox_10 = new QCheckBox(tab_stat);
        checkBox_10->setObjectName(QStringLiteral("checkBox_10"));

        gridLayout_7->addWidget(checkBox_10, 1, 1, 1, 1);

        checkBox_6 = new QCheckBox(tab_stat);
        checkBox_6->setObjectName(QStringLiteral("checkBox_6"));

        gridLayout_7->addWidget(checkBox_6, 1, 0, 1, 1);

        checkBox_8 = new QCheckBox(tab_stat);
        checkBox_8->setObjectName(QStringLiteral("checkBox_8"));

        gridLayout_7->addWidget(checkBox_8, 0, 1, 1, 1);

        checkBox_5 = new QCheckBox(tab_stat);
        checkBox_5->setObjectName(QStringLiteral("checkBox_5"));

        gridLayout_7->addWidget(checkBox_5, 0, 0, 1, 1);

        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        checkBox_9 = new QCheckBox(tab_stat);
        checkBox_9->setObjectName(QStringLiteral("checkBox_9"));

        horizontalLayout_2->addWidget(checkBox_9);

        comboBox_2 = new QComboBox(tab_stat);
        comboBox_2->setObjectName(QStringLiteral("comboBox_2"));
        comboBox_2->setIconSize(QSize(12, 12));

        horizontalLayout_2->addWidget(comboBox_2);


        gridLayout_7->addLayout(horizontalLayout_2, 2, 1, 1, 1);

        checkBox_7 = new QCheckBox(tab_stat);
        checkBox_7->setObjectName(QStringLiteral("checkBox_7"));

        gridLayout_7->addWidget(checkBox_7, 2, 0, 1, 1);

        pushButton_6 = new QPushButton(tab_stat);
        pushButton_6->setObjectName(QStringLiteral("pushButton_6"));

        gridLayout_7->addWidget(pushButton_6, 2, 2, 1, 1);

        pushButton_7 = new QPushButton(tab_stat);
        pushButton_7->setObjectName(QStringLiteral("pushButton_7"));

        gridLayout_7->addWidget(pushButton_7, 0, 2, 1, 1);

        label_8 = new QLabel(tab_stat);
        label_8->setObjectName(QStringLiteral("label_8"));

        gridLayout_7->addWidget(label_8, 1, 2, 1, 1);


        gridLayout_8->addLayout(gridLayout_7, 3, 0, 1, 1);

        tabWidget->addTab(tab_stat, QString());

        horizontalLayout->addWidget(tabWidget);

        widget = new QWidget(centralWidget);
        widget->setObjectName(QStringLiteral("widget"));

        horizontalLayout->addWidget(widget);

        horizontalLayout->setStretch(0, 3);
        horizontalLayout->setStretch(1, 2);

        verticalLayout->addLayout(horizontalLayout);

        plotLayout = new QHBoxLayout();
        plotLayout->setSpacing(6);
        plotLayout->setObjectName(QStringLiteral("plotLayout"));
        optionsPane = new QVBoxLayout();
        optionsPane->setSpacing(6);
        optionsPane->setObjectName(QStringLiteral("optionsPane"));
        label = new QLabel(centralWidget);
        label->setObjectName(QStringLiteral("label"));
        label->setScaledContents(false);

        optionsPane->addWidget(label);

        checkBox = new QCheckBox(centralWidget);
        checkBox->setObjectName(QStringLiteral("checkBox"));

        optionsPane->addWidget(checkBox);

        checkBox_3 = new QCheckBox(centralWidget);
        checkBox_3->setObjectName(QStringLiteral("checkBox_3"));

        optionsPane->addWidget(checkBox_3);

        checkBox_2 = new QCheckBox(centralWidget);
        checkBox_2->setObjectName(QStringLiteral("checkBox_2"));

        optionsPane->addWidget(checkBox_2);

        line = new QFrame(centralWidget);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::HLine);
        line->setFrameShadow(QFrame::Sunken);

        optionsPane->addWidget(line);

        label_2 = new QLabel(centralWidget);
        label_2->setObjectName(QStringLiteral("label_2"));

        optionsPane->addWidget(label_2);

        listWidget = new QListWidget(centralWidget);
        new QListWidgetItem(listWidget);
        new QListWidgetItem(listWidget);
        listWidget->setObjectName(QStringLiteral("listWidget"));

        optionsPane->addWidget(listWidget);

        line_2 = new QFrame(centralWidget);
        line_2->setObjectName(QStringLiteral("line_2"));
        line_2->setFrameShape(QFrame::HLine);
        line_2->setFrameShadow(QFrame::Sunken);

        optionsPane->addWidget(line_2);

        verticalSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);

        optionsPane->addItem(verticalSpacer);


        plotLayout->addLayout(optionsPane);

        plotPane = new QWidget(centralWidget);
        plotPane->setObjectName(QStringLiteral("plotPane"));

        plotLayout->addWidget(plotPane);

        plotLayout->setStretch(0, 1);
        plotLayout->setStretch(1, 6);

        verticalLayout->addLayout(plotLayout);

        verticalLayout->setStretch(0, 1);
        verticalLayout->setStretch(1, 5);
        latools->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(latools);
        menuBar->setObjectName(QStringLiteral("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 1221, 22));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QStringLiteral("menuFile"));
        menuEdit = new QMenu(menuBar);
        menuEdit->setObjectName(QStringLiteral("menuEdit"));
        menuProcess = new QMenu(menuBar);
        menuProcess->setObjectName(QStringLiteral("menuProcess"));
        menuHelp = new QMenu(menuBar);
        menuHelp->setObjectName(QStringLiteral("menuHelp"));
        latools->setMenuBar(menuBar);
        mainToolBar = new QToolBar(latools);
        mainToolBar->setObjectName(QStringLiteral("mainToolBar"));
        latools->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(latools);
        statusBar->setObjectName(QStringLiteral("statusBar"));
        latools->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuBar->addAction(menuEdit->menuAction());
        menuBar->addAction(menuProcess->menuAction());
        menuBar->addAction(menuHelp->menuAction());
        menuFile->addAction(actionLoad_Data_2);
        menuProcess->addAction(actionDespike);
        menuProcess->addAction(actionIdentify_Background);
        menuProcess->addAction(actionBackground_Correct);
        menuProcess->addAction(actionRatio);
        menuProcess->addAction(actionIdentify_Standards);
        menuProcess->addAction(actionApply_SRMs);

        retranslateUi(latools);

        tabWidget->setCurrentIndex(3);


        QMetaObject::connectSlotsByName(latools);
    } // setupUi

    void retranslateUi(QMainWindow *latools)
    {
        latools->setWindowTitle(QApplication::translate("latools", "latools", 0));
        actionLoad_Data->setText(QApplication::translate("latools", "Load Data", 0));
        actionLoad_Data_2->setText(QApplication::translate("latools", "Load Data", 0));
        actionDespike->setText(QApplication::translate("latools", "Despike", 0));
        actionIdentify_Background->setText(QApplication::translate("latools", "Identify Background", 0));
        actionBackground_Correct->setText(QApplication::translate("latools", "Background Correct", 0));
        actionRatio->setText(QApplication::translate("latools", "Calculate Ratios", 0));
        actionIdentify_Standards->setText(QApplication::translate("latools", "Identify Standards", 0));
        actionApply_SRMs->setText(QApplication::translate("latools", "Apply SRMs", 0));
        pushButton->setText(QApplication::translate("latools", "Import Data", 0));
        pushButton_2->setText(QApplication::translate("latools", "Despike Data", 0));
        label_3->setText(QApplication::translate("latools", "TextLabel", 0));
        label_4->setText(QApplication::translate("latools", "TextLabel", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_load), QApplication::translate("latools", "Data Import", 0));
        checkBox_4->setText(QApplication::translate("latools", "Refine ID Mode", 0));
        pushButton_3->setText(QApplication::translate("latools", "Automatic ID", 0));
        label_5->setText(QApplication::translate("latools", "TextLabel", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_bkg), QApplication::translate("latools", "Background Correction", 0));
        pushButton_5->setText(QApplication::translate("latools", "SRM Data File", 0));
        pushButton_4->setText(QApplication::translate("latools", "Identify SRMs", 0));
        label_6->setText(QApplication::translate("latools", "TextLabel", 0));
        label_7->setText(QApplication::translate("latools", "TextLabel", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_srm), QApplication::translate("latools", "Calibration", 0));
        checkBox_10->setText(QApplication::translate("latools", "Min, Max", 0));
        checkBox_6->setText(QApplication::translate("latools", "Standard Deviation", 0));
        checkBox_8->setText(QApplication::translate("latools", "Median", 0));
        checkBox_5->setText(QApplication::translate("latools", "Mean", 0));
        checkBox_9->setText(QApplication::translate("latools", "Percentile", 0));
        checkBox_7->setText(QApplication::translate("latools", "Standard Error", 0));
        pushButton_6->setText(QApplication::translate("latools", "Calculate", 0));
        pushButton_7->setText(QApplication::translate("latools", "Output File", 0));
        label_8->setText(QApplication::translate("latools", "TextLabel", 0));
        tabWidget->setTabText(tabWidget->indexOf(tab_stat), QApplication::translate("latools", "Statistics", 0));
        label->setText(QApplication::translate("latools", "Analytes", 0));
        checkBox->setText(QApplication::translate("latools", "A", 0));
        checkBox_3->setText(QApplication::translate("latools", "B", 0));
        checkBox_2->setText(QApplication::translate("latools", "C", 0));
        label_2->setText(QApplication::translate("latools", "Samples", 0));

        const bool __sortingEnabled = listWidget->isSortingEnabled();
        listWidget->setSortingEnabled(false);
        QListWidgetItem *___qlistwidgetitem = listWidget->item(0);
        ___qlistwidgetitem->setText(QApplication::translate("latools", "Sample 1", 0));
        QListWidgetItem *___qlistwidgetitem1 = listWidget->item(1);
        ___qlistwidgetitem1->setText(QApplication::translate("latools", "Sample 2", 0));
        listWidget->setSortingEnabled(__sortingEnabled);

        menuFile->setTitle(QApplication::translate("latools", "File", 0));
        menuEdit->setTitle(QApplication::translate("latools", "Edit", 0));
        menuProcess->setTitle(QApplication::translate("latools", "Process", 0));
        menuHelp->setTitle(QApplication::translate("latools", "Help", 0));
    } // retranslateUi

};

namespace Ui {
    class latools: public Ui_latools {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_LATOOLS_H
