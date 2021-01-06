#include "rule.h"


Rule::Rule(std::unique_ptr<IBody> bodyPtr, std::unique_ptr<IHead> headPtr)
    : bodyPtr_(std::move(bodyPtr)), headPtr_(std::move(headPtr)) {

}

const IBody& Rule::getBody() const {
    return *bodyPtr_;
}

const IHead& Rule::getHead() const {
    return *headPtr_;
}
