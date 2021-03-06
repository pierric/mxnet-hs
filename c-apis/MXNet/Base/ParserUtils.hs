module MXNet.Base.ParserUtils (
    parseR,
    list,
    tuple,
    module Data.Attoparsec.Text
) where

import           Control.Lens         (ix, (^?!), _Right)
import           Data.Attoparsec.Text
import qualified Data.Attoparsec.Text as P
import           RIO

parseR :: HasCallStack => P.Parser a -> Text -> a
parseR c t = (parseOnly (c <* endOfInput) t) ^?! _Right

list  f = char '[' *> commaSep f <* char ']'
tuple f = char '(' *> liftA2 (,) f (char ',' *> f) <* char ')'

-- TODO give type level int, build a tuple of that size
many f = char '(' *> commaSep f <* char ')'

commaSep p  = p `sepBy` (char ',')
