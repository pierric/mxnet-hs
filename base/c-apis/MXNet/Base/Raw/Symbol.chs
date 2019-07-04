{-# LANGUAGE ScopedTypeVariables #-}
module MXNet.Base.Raw.Symbol where

import Foreign.Marshal (alloca, withArray, peekArray, allocaBytesAligned)
import Foreign.Storable (Storable(..))
import Foreign.Ptr (FunPtr)
import Foreign.C.Types
import Foreign.C.String
import Foreign.Ptr
import Foreign.Concurrent (newForeignPtr)
import Foreign.ForeignPtr (touchForeignPtr)
import Foreign.ForeignPtr.Unsafe (unsafeForeignPtrToPtr)
import C2HS.C.Extra.Marshal (withIntegralArray, peekIntegralArray, peekString, peekStringArray)
import GHC.Generics (Generic)
import Control.Monad ((>=>))

{# import MXNet.Base.Raw.Common #}
{# import MXNet.Base.Raw.NDArray #}

#include <mxnet/c_api.h>

{# typedef size_t CSize#}
{# default in `CSize' [size_t] fromIntegral #}
{# typedef mx_uint MX_UINT#}
{# default in `MX_UINT' [mx_uint] id #}

{#
pointer SymbolHandle foreign newtype
#}

deriving instance Generic SymbolHandle

type SymbolHandlePtr = Ptr SymbolHandle

newSymbolHandle :: SymbolHandlePtr -> IO SymbolHandle
newSymbolHandle ptr = newForeignPtr ptr (mxSymbolFree ptr) >>= return . SymbolHandle

peekSymbolHandle :: Ptr SymbolHandlePtr -> IO SymbolHandle
peekSymbolHandle = peek >=> newSymbolHandle

withSymbolHandleArray :: [SymbolHandle] -> (Ptr SymbolHandlePtr -> IO r) -> IO r
withSymbolHandleArray array io = do
    let unSymbolHandle (SymbolHandle fptr) = fptr
    r <- withArray (map (unsafeForeignPtrToPtr . unSymbolHandle) array) io
    mapM_ (touchForeignPtr . unSymbolHandle) array
    return r

{#
fun MXSymbolFree as mxSymbolFree_
    {
        id `SymbolHandlePtr'
    } -> `CInt'
#}

mxSymbolFree :: SymbolHandlePtr -> IO ()
mxSymbolFree = checked . mxSymbolFree_

{#
fun MXSymbolCreateAtomicSymbol as mxSymbolCreateAtomicSymbol_
    {
        `AtomicSymbolCreator',
        `MX_UINT',
        withStringArray* `[String]',
        withStringArray* `[String]',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolCreateAtomicSymbol :: AtomicSymbolCreator
                           -> [String]
                           -> [String]
                           -> IO SymbolHandle
mxSymbolCreateAtomicSymbol creator keys vals = do
    let argcnt = fromIntegral $ length keys
    checked $ mxSymbolCreateAtomicSymbol_ creator argcnt keys vals

{#
fun MXSymbolCreateVariable as mxSymbolCreateVariable_
    {
        `String',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolCreateVariable :: String -> IO SymbolHandle
mxSymbolCreateVariable = checked . mxSymbolCreateVariable_

{#
fun MXSymbolCreateFromJSON as mxSymbolCreateFromJSON_
    {
        `String',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolCreateFromJSON :: String -> IO SymbolHandle
mxSymbolCreateFromJSON = checked . mxSymbolCreateFromJSON_

{#
fun MXSymbolSaveToJSON as mxSymbolSaveToJSON_
    {
        `SymbolHandle',
        alloca- `String' peekString*
    } -> `CInt'
#}

mxSymbolSaveToJSON :: SymbolHandle -> IO String
mxSymbolSaveToJSON = checked . mxSymbolSaveToJSON_

{#
fun MXSymbolCopy as mxSymbolCopy_
    {
        `SymbolHandle',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolCopy :: SymbolHandle -> IO SymbolHandle
mxSymbolCopy = checked . mxSymbolCopy_

{#
fun MXSymbolPrint as mxSymbolPrint_
    {
        `SymbolHandle',
        alloca- `String' peekString*
    } -> `CInt'
#}

mxSymbolPrint :: SymbolHandle -> IO String
mxSymbolPrint = checked . mxSymbolPrint_

{#
fun MXSymbolGetName as mxSymbolGetName_
    {
        `SymbolHandle',
        alloca- `Ptr (Ptr CChar)' id,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolGetName :: SymbolHandle -> IO (Maybe String)
mxSymbolGetName symbol = do
    (out, succ) <- checked $ mxSymbolGetName_ symbol
    if succ == 0 then
        Just <$> peekString out
    else
        return Nothing

{#
fun MXSymbolGetAttr as mxSymbolGetAttr_
    {
        `SymbolHandle',
        `String',
        alloca- `Ptr (Ptr CChar)' id,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolGetAttr :: SymbolHandle
                -> String
                -> IO (Maybe String)
mxSymbolGetAttr symbol key = do
    (pstr, succ) <- checked $ mxSymbolGetAttr_ symbol key
    if succ == 0 then
        Just <$> peekString pstr
    else
        return Nothing

{#
fun MXSymbolSetAttr as mxSymbolSetAttr_
    {
        `SymbolHandle',
        `String',
        `String'
    } -> `CInt'
#}

mxSymbolSetAttr :: SymbolHandle
                -> String
                -> String
                -> IO ()
mxSymbolSetAttr symbol key val = checked $ mxSymbolSetAttr_ symbol key val

{#
fun MXSymbolListAttr as mxSymbolListAttr_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr (Ptr CChar)' peek*
    } -> `CInt'
#}

mxSymbolListAttr :: SymbolHandle
                 -> IO [String]
mxSymbolListAttr symbol = do
    (cnt, ptr) <- checked $ mxSymbolListAttr_ symbol
    peekStringArray (fromIntegral cnt) ptr

{#
fun MXSymbolListAttrShallow as mxSymbolListAttrShallow_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr (Ptr CChar)' peek*
    } -> `CInt'
#}

mxSymbolListAttrShallow :: SymbolHandle
                        -> IO [String]
mxSymbolListAttrShallow symbol = do
    (cnt, ptr) <- checked $ mxSymbolListAttrShallow_ symbol
    peekStringArray (fromIntegral cnt) ptr

{#
fun MXSymbolListArguments as mxSymbolListArguments_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr (Ptr CChar)' peek*
    } -> `CInt'
#}

mxSymbolListArguments :: SymbolHandle
                      -> IO [String]
mxSymbolListArguments symbol = do
    (cnt, ptr) <- checked $ mxSymbolListArguments_ symbol
    peekStringArray (fromIntegral cnt) ptr

{#
fun MXSymbolListOutputs as mxSymbolListOutputs_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr (Ptr CChar)' peek*
    } -> `CInt'
#}

mxSymbolListOutputs :: SymbolHandle
                    -> IO [String]
mxSymbolListOutputs symbol = do
    (cnt, ptr) <- checked $ mxSymbolListOutputs_ symbol
    peekStringArray (fromIntegral cnt) ptr

{#
fun MXSymbolGetInternals as mxSymbolGetInternals_
    {
        `SymbolHandle',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolGetInternals :: SymbolHandle -> IO SymbolHandle
mxSymbolGetInternals = checked . mxSymbolGetInternals_

{#
fun MXSymbolGetChildren as mxSymbolGetChildren_
    {
        `SymbolHandle',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}

mxSymbolGetChildren :: SymbolHandle -> IO SymbolHandle
mxSymbolGetChildren = checked . mxSymbolGetChildren_

{#
fun MXSymbolGetOutput as mxSymbolGetOutput_
    {
        `SymbolHandle',
        `MX_UINT',
        alloca- `SymbolHandle' peekSymbolHandle*
    } -> `CInt'
#}
mxXSymbolGetOutput :: SymbolHandle
                   -> Int
                   -> IO SymbolHandle
mxXSymbolGetOutput symbol ind =
    checked $ mxSymbolGetOutput_ symbol (fromIntegral ind)

{#
fun MXSymbolListAuxiliaryStates as mxSymbolListAuxiliaryStates_
    {
        `SymbolHandle',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr (Ptr CChar)' peek*
    } -> `CInt'
#}
mxSymbolListAuxiliaryStates :: SymbolHandle -> IO [String]
mxSymbolListAuxiliaryStates symbol = do
    (cnt, ptr) <- checked $ mxSymbolListAuxiliaryStates_ symbol
    peekStringArray (fromIntegral cnt) ptr

{#
fun MXSymbolCompose as mxSymbolCompose_
    {
        `SymbolHandle',
        `String',
        `MX_UINT',
        id `Ptr (Ptr CChar)',
        withSymbolHandleArray* `[SymbolHandle]'
    } -> `CInt'
#}

mxSymbolCompose :: SymbolHandle
                -> String
                -> Maybe [String]
                -> [SymbolHandle]
                -> IO ()
mxSymbolCompose symbol name maybekeys args = do
    let len = fromIntegral $ length args
    case maybekeys of
      Nothing -> checked $ mxSymbolCompose_ symbol name len C2HSImp.nullPtr args
      Just keys -> withStringArray keys $ \pkeys -> checked $ mxSymbolCompose_ symbol name len pkeys args

{#
fun MXSymbolInferShape as mxSymbolInferShape_
    {
        `SymbolHandle',
        `MX_UINT',
        withStringArray* `[String]',
        withArray* `[MX_UINT]',
        withArray* `[MX_UINT]',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr MX_UINT' peek*,
        alloca- `Ptr (Ptr MX_UINT)' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr MX_UINT' peek*,
        alloca- `Ptr (Ptr MX_UINT)' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr MX_UINT' peek*,
        alloca- `Ptr (Ptr MX_UINT)' peek*,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolInferShape :: SymbolHandle
                   -> [String]
                   -> [Int]
                   -> [Int]
                   -> IO ([[Int]], [[Int]], [[Int]], Bool)
mxSymbolInferShape symbol keys arg_ind arg_shape = do
    let num_args' = fromIntegral (length keys)
        arg_ind'  = map fromIntegral arg_ind
        arg_shape'= map fromIntegral arg_shape
    (inshape_size , inshape_ndim , inshape_data ,
     outshape_size, outshape_ndim, outshape_data,
     auxshape_size, auxshape_ndim, auxshape_data,
     complete) <- checked $ mxSymbolInferShape_ symbol num_args' keys arg_ind' arg_shape'
    let inshape_size' = fromIntegral inshape_size
    inshape_ndim'    <- peekArray inshape_size' inshape_ndim
    inshape_data'    <- peekArray inshape_size' inshape_data
    inshape_data_ret <- mapM (uncurry peekArrayOfUInt) (zip inshape_ndim' inshape_data')
    let outshape_size'= fromIntegral outshape_size
    outshape_ndim'   <- peekArray outshape_size' outshape_ndim
    outshape_data'   <- peekArray outshape_size' outshape_data
    outshape_data_ret<- mapM (uncurry peekArrayOfUInt) (zip outshape_ndim' outshape_data')
    let auxshape_size'= fromIntegral auxshape_size
    auxshape_ndim'   <- peekArray auxshape_size' auxshape_ndim
    auxshape_data'   <- peekArray auxshape_size' auxshape_data
    auxshape_data_ret<- mapM (uncurry peekArrayOfUInt) (zip auxshape_ndim' auxshape_data')
    return (inshape_data_ret, outshape_data_ret, auxshape_data_ret, complete == 1)

{#
fun MXSymbolInferShapePartial as mxSymbolInferShapePartial_
    {
        `SymbolHandle',
        `MX_UINT',
        withStringArray* `[String]',
        withArray* `[MX_UINT]',
        withArray* `[MX_UINT]',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr MX_UINT' peek*,
        alloca- `Ptr (Ptr MX_UINT)' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr MX_UINT' peek*,
        alloca- `Ptr (Ptr MX_UINT)' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr MX_UINT' peek*,
        alloca- `Ptr (Ptr MX_UINT)' peek*,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolInferShapePartial :: SymbolHandle
                   -> [String]
                   -> [Int]
                   -> [Int]
                   -> IO ([[Int]], [[Int]], [[Int]], Bool)
mxSymbolInferShapePartial symbol keys arg_ind arg_shape = do
    let num_args' = fromIntegral (length keys)
        arg_ind'  = map fromIntegral arg_ind
        arg_shape'= map fromIntegral arg_shape
    (inshape_size , inshape_ndim , inshape_data ,
     outshape_size, outshape_ndim, outshape_data,
     auxshape_size, auxshape_ndim, auxshape_data,
     complete) <- checked $ mxSymbolInferShapePartial_ symbol num_args' keys arg_ind' arg_shape'
    let inshape_size' = fromIntegral inshape_size
    inshape_ndim'    <- peekArray inshape_size' inshape_ndim
    inshape_data'    <- peekArray inshape_size' inshape_data
    inshape_data_ret <- mapM (uncurry peekArrayOfUInt) (zip inshape_ndim' inshape_data')
    let outshape_size'= fromIntegral outshape_size
    outshape_ndim'   <- peekArray outshape_size' outshape_ndim
    outshape_data'   <- peekArray outshape_size' outshape_data
    outshape_data_ret<- mapM (uncurry peekArrayOfUInt) (zip outshape_ndim' outshape_data')
    let auxshape_size'= fromIntegral auxshape_size
    auxshape_ndim'   <- peekArray auxshape_size' auxshape_ndim
    auxshape_data'   <- peekArray auxshape_size' auxshape_data
    auxshape_data_ret<- mapM (uncurry peekArrayOfUInt) (zip auxshape_ndim' auxshape_data')
    return (inshape_data_ret, outshape_data_ret, auxshape_data_ret, complete == 1)

peekArrayOfUInt cnt ptr = peekIntegralArray (fromIntegral cnt)  ptr

{#
fun MXSymbolInferType as mxSymbolInferType_
    {
        `SymbolHandle',
        `MX_UINT',
        withStringArray* `[String]',
        withArray* `[CInt]',
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `MX_UINT' peek*,
        alloca- `Ptr CInt' peek*,
        alloca- `CInt' peek*
    } -> `CInt'
#}

mxSymbolInferType :: SymbolHandle
                  -> [String]
                  -> [Int]
                  -> IO (Maybe ([Int], [Int], [Int]))
mxSymbolInferType symbol keys arg_type = do
    let num_args = fromIntegral (length keys)
        arg_type'= map fromIntegral arg_type
    (inshape_size , inshape_data ,
     outshape_size, outshape_data,
     auxshape_size, auxshape_data,
     succ) <- checked $ mxSymbolInferType_ symbol num_args keys arg_type'
    if succ == 0 then do
        inshape_data_ret  <- peekIntegralArray (fromIntegral inshape_size)  inshape_data
        outshape_data_ret <- peekIntegralArray (fromIntegral outshape_size) outshape_data
        auxshape_data_ret <- peekIntegralArray (fromIntegral auxshape_size) auxshape_data
        return $ Just (inshape_data_ret, outshape_data_ret, auxshape_data_ret)
    else
        return Nothing

{#
fun MXSymbolSaveToFile as mxSymbolSaveToFile_
    {
        `SymbolHandle',
        `String'
    } -> `CInt'
#}

mxSymbolSaveToFile :: String -> SymbolHandle -> IO ()
mxSymbolSaveToFile filename sym = do
    checked $ mxSymbolSaveToFile_ sym filename

data MXCallbackList = MXCallbackList Int (Ptr (FunPtr (IO CInt))) (Ptr (Ptr ()))

instance Storable MXCallbackList where
    sizeOf _ = {#sizeof MXCallbackList#}
    alignment _ = {#alignof MXCallbackList#}
    peek ptr = do
        n <- {#get MXCallbackList.num_callbacks#} ptr
        callbacks <- {#get MXCallbackList.callbacks#} ptr
        contexts  <- {#get MXCallbackList.contexts#} ptr
        return (MXCallbackList (fromIntegral n) callbacks contexts)
    poke ptr (MXCallbackList n callbacks contexts) = do
        {#set MXCallbackList.num_callbacks#} ptr (fromIntegral n)
        {#set MXCallbackList.callbacks#} ptr callbacks
        {#set MXCallbackList.contexts#}  ptr contexts

type CustomOpPropCreator          = CString -> CInt -> Ptr CString -> Ptr CString -> Ptr () -> IO CInt
type CustomOpFBFunc               = CInt -> Ptr (Ptr ()) -> Ptr CInt -> Ptr CInt -> CInt -> Ptr () -> IO CInt
type CustomOpDelFunc              = Ptr () -> IO CInt
type CustomOpListFunc             = Ptr (Ptr CString) -> Ptr () -> IO CInt
type CustomOpInferShapeFunc       = CInt -> Ptr CInt -> Ptr (Ptr CInt) -> Ptr () -> IO CInt
type CustomOpInferStorageTypeFunc = CInt -> Ptr CInt -> Ptr () -> IO CInt
type CustomOpBackwardInferStorageTypeFunc = CInt -> Ptr CInt -> Ptr CInt -> Ptr () -> IO CInt
type CustomOpInferTypeFunc        = CInt -> Ptr CInt -> Ptr () -> IO CInt
type CustomOpBwdDepFunc           = Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr CInt -> Ptr (Ptr CInt) -> Ptr () -> IO CInt
type CustomOpCreateFunc           = CString -> CInt -> Ptr (Ptr CUInt) -> Ptr CInt -> Ptr CInt -> Ptr MXCallbackList -> Ptr () -> IO CInt
type CustomFunctionBwdFunc        = CInt -> Ptr (Ptr NDArrayHandle) -> Ptr CInt -> Ptr CInt -> CInt -> Ptr () -> IO CInt
type CustomFunctionDelFunc        = Ptr () -> IO CInt

{#
fun MXCustomOpRegister as mxCustomOpRegister_
    {
        `String',
        id `FunPtr CustomOpPropCreator'
    } -> `CInt'
#}

mxCustomOpRegister :: String -> FunPtr CustomOpPropCreator -> IO ()
mxCustomOpRegister op cr = checked $ mxCustomOpRegister_ op cr

-- {#
-- fun as
--     {
--
--     } -> `CInt'
-- #}

-- {#
-- fun MXSymbolGetInputSymbols as mxSymbolGetInputSymbols_
--     {
--         `SymbolHandle',
--         alloca- `Ptr SymbolHandlePtr' peek*,
--         alloca- `CInt' peek*
--     } -> `CInt'
-- #}
--
-- {#
-- fun MXSymbolGetNumOutputs as mxSymbolGetNumOutputs_
--     {
--         `SymbolHandle',
--         alloca- `MX_UINT' peek*
--     } -> `CInt'
-- #}
--
-- mxSymbolGetNumOutputs :: SymbolHandle -> IO Int
-- mxSymbolGetNumOutputs symbol = do
--     cnt <- checked $ mxSymbolGetNumOutputs_ symbol
--     return $ fromIntegral cnt
--
-- {#
-- fun MXQuantizeSymbol as mxQuantizeSymbol_
--     {
--         `SymbolHandle',
--         alloca- `Ptr SymbolHandlePtr' peekSymbolHandle*,
--         `MX_UINT',
--         withSymbolHandleArray* `[SymbolHandle]',
--         `MX_UINT',
--         withStringArray* `[String]',
--         `String'
--     } -> `CInt'
-- #}
