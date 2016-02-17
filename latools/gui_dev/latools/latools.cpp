#include "latools.h"
#include "ui_latools.h"

latools::latools(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::latools)
{
    ui->setupUi(this);
}

latools::~latools()
{
    delete ui;
}
