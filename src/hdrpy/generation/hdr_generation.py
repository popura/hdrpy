class DebevecMerge:
    def __init__(self, args):
        self.img_fn = args.imgs
        self.exposure_times = np.array(args.exposure_times, dtype=np.float32)

        if len(self.img_fn) != len(self.exposure_times):
            sys.stderr.write('List Size Error!')

        self.img_list = self.readImg()

    def DebevecHDR(self):
        merge_debvec = cv2.createMergeDebevec()
        self.hdr_debvec = merge_debvec.process(self.img_list, times=self.exposure_times.copy())
        return self.hdr_debvec
