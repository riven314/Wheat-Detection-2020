from fastai2.vision.all import Learner, CancelBatchException


class WheatLearner(Learner):
    def one_batch(self, i, b):
        self.iter = i
        try:
            self._split(b);                                  self('begin_batch')
            
            # model(self.xb, self.yb) = loss
            self.pred = self.model(*self.xb, *self.yb);     self('after_pred')
                
            if len(self.yb) == 0: return
            
            self.loss = self.loss_func(self.pred, *self.yb);self('after_loss')
            if not self.training: return
            self.loss.backward();                            self('after_backward')
            self.opt.step();                                 self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:                        self('after_cancel_batch')
        finally:                                            self('after_batch')