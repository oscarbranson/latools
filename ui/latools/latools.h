#ifndef LATOOLS_H
#define LATOOLS_H

#include <QMainWindow>

namespace Ui {
class latools;
}

class latools : public QMainWindow
{
    Q_OBJECT

public:
    explicit latools(QWidget *parent = 0);
    ~latools();

private:
    Ui::latools *ui;
};

#endif // LATOOLS_H
