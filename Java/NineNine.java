package hello;

public class NineNine {
	public static void main(String[] args) {
		for(int i=1;i<=9;i++) {
			for(int l=1;l<=i;l++) {
				System.out.print(l+"*"+i+"="+(l*i)+"  ");
			}
			System.out.println();
		}
	}
}
