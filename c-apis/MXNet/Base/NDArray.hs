{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.NDArray where

import Foreign.Ptr (castPtr)
import Foreign.Storable (Storable(..))
import Data.Vector.Storable (Vector)
import qualified Data.Vector.Storable as V
import qualified Data.Vector.Storable.Mutable as VMut
import qualified Data.Vector.Unboxed as UV
import qualified Data.Array.Repa as Repa
import qualified Data.Array.Repa.Eval as Repa

import qualified MXNet.Base.Raw as I
import MXNet.Base.Types (Context(..), contextCPU, DType(..), ForeignData(..))

newtype NDArray a = NDArray { unNDArray :: I.NDArrayHandle}

instance ForeignData (NDArray a) where
    touch = I.touchNDArrayHandle . unNDArray

makeEmptyNDArray :: forall a. DType a => [Int] -> Context -> IO (NDArray a)
makeEmptyNDArray shape ctx = do
    array <- I.mxNDArrayCreateEx shape (_device_type ctx) (_device_id ctx) False (flag (undefined :: a))
    return $ NDArray array

makeNDArray :: forall a. DType a => [Int] -> Context -> Vector a -> IO (NDArray a)
makeNDArray shape ctx vec = do
    array <- I.mxNDArrayCreateEx shape (_device_type ctx) (_device_id ctx) False (flag (undefined :: a))
    V.unsafeWith vec $ \p -> do
        I.mxNDArraySyncCopyFromCPU array (castPtr p) (V.length vec)
        return $ NDArray array

makeNDArrayLike :: DType a => NDArray a -> Context -> IO (NDArray a)
makeNDArrayLike src cxt = do
    shape <- ndshape src
    makeEmptyNDArray shape cxt
        
ndshape :: DType a => NDArray a -> IO [Int]
ndshape = I.mxNDArrayGetShape . unNDArray

ndsize :: DType a => NDArray a -> IO Int
ndsize arr = product <$> ndshape arr

full :: DType a => a -> [Int] -> IO (NDArray a)
full value shape = makeNDArray shape contextCPU $ V.replicate (product shape) value

ones :: DType a => [Int] -> IO (NDArray a)
ones = full 1

zeros :: DType a => [Int] -> IO (NDArray a)
zeros = full 1

fromVector :: DType a => [Int] -> Vector a -> IO (NDArray a)
fromVector shape = makeNDArray shape contextCPU

copyFromVector :: DType a => NDArray a -> Vector a -> IO ()
copyFromVector arr vec = do
    sz <- ndsize arr
    if (sz /= V.length vec) 
      then error ""
      else do
        V.unsafeWith vec $ \p -> do
            I.mxNDArraySyncCopyFromCPU (unNDArray arr) (castPtr p) sz

copyFromRepa :: (Repa.Shape sh, DType a, UV.Unbox a, Repa.Load r sh a) => NDArray a -> Repa.Array r sh a -> IO ()
copyFromRepa arr repa = do
    let vec = UV.convert $ Repa.toUnboxed $ Repa.computeS repa
    copyFromVector arr vec

toVector :: DType a => NDArray a -> IO (Vector a)
toVector arr = do
    nlen <- ndsize arr
    mvec <- VMut.new nlen
    VMut.unsafeWith mvec $ \p -> I.mxNDArraySyncCopyToCPU (unNDArray arr) (castPtr p) nlen
    V.unsafeFreeze mvec

toRepa :: (Repa.Shape sh, DType a, UV.Unbox a) => NDArray a -> IO (Repa.Array Repa.U sh a)
toRepa arr = do
    shp <- ndshape arr
    vec <- toVector arr
    return $ Repa.fromUnboxed (Repa.shapeOfList (reverse shp)) (UV.convert vec)

context :: DType a => NDArray a -> IO Context
context (NDArray handle) = do
    cxt <- I.mxNDArrayGetContext handle
    return $ uncurry Context cxt

waitToRead :: DType a => NDArray a -> IO ()
waitToRead (NDArray hdl) = I.mxNDArrayWaitToRead hdl

waitToWrite :: DType a => NDArray a -> IO ()
waitToWrite (NDArray hdl) = I.mxNDArrayWaitToWrite hdl

waitAll = I.mxNDArrayWaitAll